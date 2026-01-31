import numpy as np
import skimage as ski
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import pixelart_lib.basics as bsc

def create_palette(
    image: np.ndarray,
    n_colors: int = 8,
    ):
    """
    Create a reduced colour palette of `n` colours.

    Parameters
    ----------
    image : np.ndarray
        RGB- (or RGBA-)Image to choose the colours from. 
    n : int
        Number of colours to choose. Defaults to 8.

    Returns
    -------
    rgb_palette : 
        A reduced colour palettte based on the coulours
        in `image`
    """
    # Test if there is an alpha channel and drop it
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Convert to LAB color space
    image_lab = ski.color.rgb2lab(image)

    # Convert the image to a list of pixels
    pixels = image_lab.reshape(-1, 3)

    # Fit a KMeans clustering model to the data
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(pixels)

    # Get the colors of the clusters
    lab_palette = kmeans.cluster_centers_

    # Convert the palette to RGB
    rgb_palette = ski.color.lab2rgb(lab_palette[np.newaxis, :])[0]

    return rgb_palette

def pixelate_image(image, block_size, mask, palette):
    """
    Pixelate parts of the image that are not masked.

    Parameters:
    - image: Input color image as a NumPy array.
    - block_size: Size of the blocks (e.g., 8 for 8x8 blocks).
    - mask: Boolean mask of the same shape as the image, where `True` means the region
            should NOT be pixelated, and `False` means it should be pixelated.

    Returns:
    - pixelated_image: The partially pixelated image.
    """
    # Ensure the mask is 2D (height, width)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Pad the image to make dimensions divisible by block_size
    padded_image, pad_h, pad_w = pad_to_multiple(image, block_size)
    #show_image(ski.color.lab2rgb(padded_image), title='Padded image')

    small_padded_image = reduce_resolution(padded_image, block_size)
    #show_image(ski.color.lab2rgb(small_padded_image), title='Small padded image')

    quantised_small_padded_image = map_to_palette(small_padded_image, palette)
    #show_image(ski.color.lab2rgb(quantised_small_padded_image), title='Quantised small padded image')

    upscaled_padded_image = upscale_image(quantised_small_padded_image, block_size)
    #show_image(ski.color.lab2rgb(upscaled_padded_image), title='Upscaled padded image')

    pixelated = crop_to_original(upscaled_padded_image, image.shape[0], image.shape[1])
    pixelated = ski.color.lab2rgb(pixelated)
    #show_image(pixelated, title='Pixelated')

    # Apply mask to combine original and pixelated images
    original_card = ski.color.lab2rgb(image.copy())
    pixelated[~mask] = original_card[~mask]

    return pixelated

def map_to_palette(image, palette):
    '''
    Map the colors of the image to the colors in the palette.

    Parameters:
    - image: Input color image as a NumPy array using LAB as colour space.
    - palette: List of colors in LAB colour space to map the image to.

    Returns:
    - new_image: The image with its colors mapped to the palette.
    '''
    # Drop the alpha channel if it exists
    if image.shape[2] == 4:
        image_lab = image[:, :, :3]
    else:
        image_lab = image

    # Convert the image to a list of pixels
    pixels = image_lab.reshape(-1, 3)

    # Create a KD-tree data structure for fast nearest-neighbor lookup
    tree = KDTree(palette)

    # Find the nearest color in the palette for each pixel
    _, indices = tree.query(pixels)
    new_pixels = palette[indices]

    # Convert the pixel list back to an image
    new_image = new_pixels.reshape(image_lab.shape)

    if image.shape[2] == 4:
        new_image = np.dstack((new_image, image[:,:, 3]/255.0))

    return new_image

def reduce_resolution(image, block_size):
    """
    Downscale the image resolution using nearest-neighbor interpolation.
    """
    height, width = image.shape[:2]
    small_image = ski.transform.resize(
        image, 
        (height // block_size, width // block_size), 
        order=0,  # Nearest-neighbor interpolation
        anti_aliasing=False,
        preserve_range=True
    )
    return small_image

def upscale_image(image, scale_factor):
    """
    Upscale the image to its original resolution using nearest-neighbor interpolation.
    """
    height, width = image.shape[:2]
    upscaled_image = ski.transform.resize(
        image, 
        (height * scale_factor, width * scale_factor), 
        order=0,  # Nearest-neighbor interpolation
        anti_aliasing=False,
        preserve_range=True
    )
    return upscaled_image

def pad_to_multiple(image, multiple):
    """
    Pads the image to make its dimensions divisible by a given multiple.
    """
    height, width = image.shape[:2]
    pad_h = (multiple - (height % multiple)) % multiple
    pad_w = (multiple - (width % multiple)) % multiple
    padded_image = np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),  # Pad height, width, no padding for channels
        mode='constant',
        constant_values=0  # Black padding
    )
    return padded_image, pad_h, pad_w

def crop_to_original(image, original_height, original_width):
    """
    Crops the padded image back to its original dimensions.
    """
    return image[:original_height, :original_width]

def pixelise_card(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    n_colours: int = 12,
    size_px: int = 8,
) -> np.ndarray:
    """
    Pixelates an area of `image`. The area is defined by `mask`.

    Parameters
    ----------
    image : np.ndarray
        Image to pixelate parts of. Has to be an rgb-image with
        dtype "uint8" and values in [0 255].
    mask: np.ndarray
        Binary mask of the same size as the image. True in the
        areas that are pixelated.
    n_colours : int
        Number of colours to reduce the colour palette of the image to.
    size_px : int
        Edge length of the new pixels in number of pixels of `image`.
    
    Returns
    -------
    np.ndarray
        An rgb-image where parts are pixelated.
    """
    art = bsc.extract_area(image, mask)
    palette = create_palette(art, n_colours)
    lab_palette = ski.color.rgb2lab(palette)
    lab_image_original = ski.color.rgb2lab(image[:,:, :3])
    pixel_image = pixelate_image(lab_image_original, size_px, mask, lab_palette)
    return pixel_image


