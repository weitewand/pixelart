import numpy as np

def extract_area(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Extract a area/areas specified by `mask` from the image.

    Parameters
    ----------
    image : np.ndarray
        Image to isolate the areas from.
    mask: np.ndarray
        Binary mask of the same size as the image. True in the
        areas that stay.

    Returns
    -------
    np.ndarray
        An image where only the areas specified by the mask are
        left and everything else is white.
    """
    img = image.copy()
    img[~mask] = [255, 255, 255]
    return img.astype("uint8")
