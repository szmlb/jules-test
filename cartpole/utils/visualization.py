import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms # Added import

class CartPoleVisualizer:
    def __init__(self, world_width=5.0, pole_length=1.0):
        self.world_width = world_width
        self.pole_visual_length = pole_length # Store the visual length of the pole
        self.fig, self.ax = plt.subplots()
        self.cart_width = 0.5  # Visual width of the cart
        self.cart_height = 0.25 # Visual height of the cart
        self.pole_width = 0.1   # Visual width of the pole (thicker for visibility)

        self.viewer_setup = False
        self.cart_patch = None
        self.pole_patch = None
        self.axle_dot = None

    def _setup_viewer(self):
        self.ax.set_xlim(-self.world_width / 2, self.world_width / 2)
        # Adjust ylim based on pole's visual length and cart height
        self.ax.set_ylim(-0.1, self.pole_visual_length * 1.2 + self.cart_height)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        self.ax.set_xlabel("Cart Position")
        self.ax.set_title("CartPole Visualization")

        # Cart: origin is bottom-left
        # Cart rides on y=0 line. Its bottom-left will be (cart_x - cart_width/2, 0)
        self.cart_patch = patches.Rectangle((0, 0), self.cart_width, self.cart_height, fc='blue')
        self.ax.add_patch(self.cart_patch)

        # Pole: origin is bottom-left. Define it as if its base is at (0,0) before transformations.
        # Its height is self.pole_visual_length
        self.pole_patch = patches.Rectangle((0, 0), self.pole_width, self.pole_visual_length, fc='brown', alpha=0.8)
        self.ax.add_patch(self.pole_patch)

        # Axle (pivot point on top of the cart)
        self.axle_dot = patches.Circle((0,0), 0.05, fc='black') # Radius of axle
        self.ax.add_patch(self.axle_dot)

        plt.ion() # Turn on interactive mode
        plt.show(block=False) # Show plot without blocking
        self.viewer_setup = True

    def render(self, cart_x, pole_theta):
        if not self.viewer_setup:
            self._setup_viewer()

        # Cart position
        # cart_x is the center of the cart. The cart rides on the line y=0.
        # Bottom-left corner of the cart:
        cart_bottom_left_x = cart_x - self.cart_width / 2
        cart_bottom_left_y = 0 # Cart is on the ground
        self.cart_patch.set_xy((cart_bottom_left_x, cart_bottom_left_y))

        # Pole position
        # Pole pivots on top-center of the cart.
        pole_pivot_x = cart_x
        pole_pivot_y = self.cart_height

        # Update axle (pivot) position
        self.axle_dot.center = (pole_pivot_x, pole_pivot_y)

        # The pole_patch is defined with its bottom-left at (0,0) and height self.pole_visual_length.
        # We want its bottom-center to be at (pole_pivot_x, pole_pivot_y).
        # So, its bottom-left before rotation is (pole_pivot_x - self.pole_width / 2, pole_pivot_y).
        # The pole_theta from env: 0 is upright, positive is clockwise.
        # Matplotlib's Affine2D().rotate_around rotates counter-clockwise. So use -pole_theta.

        # Set the position of the pole's bottom-left corner (as if it's not rotated)
        self.pole_patch.set_xy((pole_pivot_x - self.pole_width / 2, pole_pivot_y))

        # Create a transform: rotate around the pivot point, then add existing data transform
        transform = mtransforms.Affine2D().rotate_around(pole_pivot_x, pole_pivot_y, -pole_theta) + self.ax.transData
        self.pole_patch.set_transform(transform)

        self.fig.canvas.draw_idle() # More efficient than draw() for frequent updates
        self.fig.canvas.flush_events()
        plt.pause(0.0001) # Small pause to allow GUI to update

    def close(self):
        if self.viewer_setup:
            plt.ioff() # Turn off interactive mode
            plt.close(self.fig)
            self.viewer_setup = False

if __name__ == '__main__':
    # Example Usage
    visualizer = CartPoleVisualizer(world_width=4.8, pole_length=1.0)

    # Simulate some movement
    # Initial state (upright)
    visualizer.render(cart_x=0.0, pole_theta=0.0)
    plt.pause(1)

    # Pole tilted right
    visualizer.render(cart_x=0.5, pole_theta=0.2) # 0.2 radians right
    plt.pause(1)

    # Pole tilted left, cart moved
    visualizer.render(cart_x=-0.3, pole_theta=-0.3)
    plt.pause(1)

    # Pole far right
    visualizer.render(cart_x=1.0, pole_theta=math.pi/4) # 45 degrees right
    plt.pause(1)

    visualizer.close()
    print("Visualization example finished.")
