import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import pickle

def gaussian_splat(x, y, z, intensity, grid_size, sigma):
    """
    Generate a Gaussian splat.
    
    Parameters:
        x, y, z (float): Coordinates of the splat center.
        intensity (float): Intensity of the splat.
        grid_size (int): Size of the grid.
        sigma (float): Standard deviation of the Gaussian function.
    
    Returns:
        ndarray: 2D array representing the Gaussian splat.
    """
    xx, yy = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    d = np.sqrt((xx - x)**2 + (yy - y)**2)
    return intensity * np.exp(-d**2 / (2.0 * sigma**2))


def render_depth_image(camera_position, splats, grid_size):
    """
    Render depth image from camera view.
    
    Parameters:
        camera_position (tuple): Camera position (x, y, z).
        splats (list): List of splats, each defined as (x, y, z, intensity).
        grid_size (int): Size of the grid.
    
    Returns:
        ndarray: 2D array representing the depth image.
    """
    depth_image = np.zeros((grid_size, grid_size))
    for splat in splats:
        x, y, z, intensity = splat
        depth = np.sqrt((camera_position[0] - x)**2 + (camera_position[1] - y)**2 + (camera_position[2] - z)**2)
        depth_image += gaussian_splat(x, y, z, intensity, grid_size, sigma=1.0) / depth
    return depth_image


def load_checkpoint(checkpoint_path):
    """
    Load NerfStudio checkpoint and extract camera parameters and splats.
    
    Parameters:
        checkpoint_path (str): Path to the NerfStudio checkpoint.
    
    Returns:
        tuple: Tuple containing camera_position, splats, and grid_size.
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    camera_position = checkpoint['camera_position']
    splats = checkpoint['splats']
    grid_size = checkpoint['grid_size']
    
    return camera_position, splats, grid_size


def main(checkpoint_path):

    # Load checkpoint
    camera_position, splats, grid_size = load_checkpoint(checkpoint_path)

    # Render depth image
    depth_image = render_depth_image(camera_position, splats, grid_size)

    # Display depth image
    plt.imshow(depth_image, cmap='gray')
    plt.title('Depth Image')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main("./outputs/glass_tilted/splatfacto/2024-02-23_032900/nerfstudio_models/step-000029999.ckpt")

