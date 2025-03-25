import torch
import numpy as np
import cv2

def calculate_erosion_kernal(radius):
    """
    Calculate the size of the erosion kernel based on the camera's distance from the object.

    The erosion kernel determines the 'harshness' of the edge removal during depth map processing.
    Closer camera is to the object, the 'harsher' the kernel needs to be

    Returns:
        int: The size of the erosion kernel. Possible values are 3, 5, or 7 depending on the camera's radius.
    """
    # Calculate erosion kernal (this determines the 'harshness' of the edge removal)
    # Closer camera is to the object, the 'harsher' the kernel needs to be
    erode_kernal_size = 7
    if radius < 2.0:
        erode_kernal_size = 3
    elif radius < 5.0:
        erode_kernal_size = 5

    return erode_kernal_size

def normalise_depth_maps(depth_map, depth_background_cutoff = 0.97, depth_intensity_scale = 1.75, min_depth_intensity = 0.1, radius=4):
    """
    This function normalises depth maps by identifying and separating the background from the foreground,
    rescaling the depth intensities, and applying dilation and erosion to reduce noise and remove erroneous edges.

    Args:
        depth_maps (torch.Tensor): The input depth maps to be normalized and processed.
        depth_background_cutoff: The value that determines if the depth is part of the background
        depth_intensity_scale: Scale the object depth for better appearance and distinction from the background
        min_depth_intensity: The minimum depth intensity for pixels that are part of the object (ensures that parts of the object do not blend into the background)
        radius: distance from the camera to the origin

    Returns:
        torch.Tensor: The processed and normalized depth maps with refined depth information.
    """

    depth_np = depth_map.detach().cpu().numpy()

    # Determine background from foreground
    depth_np = np.round(depth_np, 2)
    unique, counts = np.unique(depth_np, return_counts=True)
    start_bin = unique[1]
    end_bin = unique[-1]

    # Invert image and ensure that background is still set to 0
    normalised_depths =  1.0 - ((np.maximum((depth_np-start_bin), np.full(depth_np.shape, 0)))/(end_bin-start_bin))
    normalised_depths = np.where((normalised_depths < depth_background_cutoff), normalised_depths, 0)

    # Rescale the image to ensure that depth of object has a minumum intensity
    normalised_depths = np.minimum(((normalised_depths + min_depth_intensity) * depth_intensity_scale), np.full(depth_np.shape, 1))
    normalised_depths = np.where((normalised_depths > (min_depth_intensity * depth_intensity_scale)), normalised_depths, 0)

    eroded_depths = np.array([])
    for normalised_depth in normalised_depths:

        # Remove erroneous edges from depth of object
        erode_kernal_size = calculate_erosion_kernal(radius)

        kernel = np.ones((erode_kernal_size, erode_kernal_size), np.uint8)
        normalised_depth = cv2.erode(normalised_depth, kernel, iterations=1)

        normalised_depth = np.expand_dims(normalised_depth, axis=0)

        if len(eroded_depths) == 0:
            eroded_depths = normalised_depth
        else:
            eroded_depths = np.concatenate((eroded_depths, normalised_depth))

    return torch.from_numpy(eroded_depths)