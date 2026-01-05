import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _(mo):
    import skimage as ski
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.cluster import KMeans
    from scipy.spatial import KDTree

    def show_image(image, title=None, size=4):
        return mo.image(image, caption=title, width=size*100)

    def plot_toh(image, title=None):
        plt.imshow(image, cmap='gray')
        if title:
            plt.title(title)
        plt.show()

    def crop_to_rectangular_mask(image, mask):
        # Find the bounding box of the mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return image[rmin:rmax, cmin:cmax]

    def show_image_with_dot(image, dot, title=None, size=4):
        plt.figure(figsize=(size, size)) 
        plt.imshow(image, cmap='gray')
        plt.plot(dot[1], dot[0], 'rx')
        plt.title(title)
        plt.show()

    def create_palette(image, n_colors):
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

    def show_palette(palette):
        plt.imshow(palette[np.newaxis, :, :])
        # Add numbers to the palette colors:
        for i, color in enumerate(palette):
            plt.text(i, -1, str(i), ha='center', va='center')
            plt.text(i, 1, str(round(color[0]*255)), ha='center', va='center')
            plt.text(i, 2, str(round(color[1]*255)), ha='center', va='center')
            plt.text(i, 3, str(round(color[2]*255)), ha='center', va='center')
        plt.axis('off')
        plt.show()

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
    return (
        create_palette,
        np,
        os,
        pixelate_image,
        plt,
        show_image,
        show_image_with_dot,
        show_palette,
        ski,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load image
    """)
    return


@app.cell
def _(os, show_image, ski):
    path_to_images = os.path.join(os.getcwd(), 'images', 'm3c-14-omo-queen-of-vesuva.png')
    image_original = ski.io.imread(path_to_images)
    image_original = ski.util.img_as_float(image_original)
    image_rgb = ski.color.rgba2rgb(image_original)
    image_gray = ski.color.rgb2gray(image_rgb)
    [height, width] = image_gray.shape
    show_image(image_original, 'Original Image', size=3)
    #show_image(image_rgb, 'RGB Image')
    #show_image(image_gray, 'Gray Image')

    return image_gray, image_original, image_rgb


@app.cell
def _(image_rgb):
    type(image_rgb)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Preprocessing

    ### Thresholding
    """)
    return


@app.cell
def _(image_gray, plt):
    # plot the histogram of the image
    plt.hist(image_gray.ravel(), bins=256, range=(0.0, 1.0))
    plt.title('Histogram of the image')
    #plt.show()
    return


@app.cell
def _(image_gray, show_image, ski):
    threshold = ski.filters.threshold_otsu(image_gray)
    print('Threshold:', threshold)
    binary_image = image_gray > .2
    show_image(binary_image, 'Binary Image', size=10)
    return (binary_image,)


@app.cell
def _(binary_image, show_image, ski):
    cleaned = ski.morphology.remove_small_objects(binary_image, min_size=1000)
    show_image(cleaned, 'Remove Small Objects', size=10)
    return (cleaned,)


@app.cell
def _(cleaned, show_image, ski):
    cleaned_1 = ski.morphology.remove_small_holes(cleaned, area_threshold=1000)
    show_image(cleaned_1, 'Remove Small Holes', size=4)
    return (cleaned_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Flood fill
    """)
    return


@app.cell
def _(cleaned_1, show_image_with_dot):
    seed = (45, 50)
    show_image_with_dot(cleaned_1, seed, 'Seed Point', size=10)
    return (seed,)


@app.cell
def _(cleaned_1, seed, show_image, ski):
    filled_title_bar = ski.segmentation.flood(cleaned_1, seed)
    filled_title_bar = ski.morphology.remove_small_holes(filled_title_bar, area_threshold=100000)
    show_image(filled_title_bar, 'Filled Title Bar', size=10)
    filled_title_bar.shape
    return (filled_title_bar,)


@app.cell
def _(cleaned_1, show_image_with_dot):
    seed_1 = (585, 50)
    show_image_with_dot(cleaned_1, seed_1, 'Seed Point', size=10)
    return (seed_1,)


@app.cell
def _(cleaned_1, np, show_image):
    lower_bound_text_box = np.where(cleaned_1[:, 100])
    lower_bound_text_box = lower_bound_text_box[0][-1]
    cleaned_1[lower_bound_text_box:, :] = 1
    right_bound_text_box = np.where(cleaned_1[650, :])
    right_bound_text_box = right_bound_text_box[0][-1]
    cleaned_1[800:, right_bound_text_box] = 1
    show_image(cleaned_1, 'Remove the lower bound of the text box', size=10)
    return


@app.cell
def _(cleaned_1, seed_1, show_image, ski):
    filled_text_box = ski.segmentation.flood(cleaned_1, seed_1)
    filled_text_box = ski.morphology.remove_small_holes(filled_text_box, area_threshold=300000)
    filled_text_box = ski.morphology.remove_small_objects(filled_text_box, min_size=300000)
    show_image(filled_text_box, 'Filled Text Box')
    filled_text_box.shape
    return (filled_text_box,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create Mask
    """)
    return


@app.cell
def _(filled_text_box, filled_title_bar, image_gray, np, show_image):
    filled = np.logical_or(filled_title_bar, filled_text_box)
    mask = np.ones(image_gray.shape, dtype=np.uint8)
    mask[filled] = 0
    mask = mask.astype(bool)
    show_image(mask, 'Mask')
    return (mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pixelate it!
    """)
    return


@app.cell
def _(create_palette, image_original, show_palette):
    palette = create_palette(image_original, 6)
    show_palette(palette)
    return (palette,)


@app.cell
def _(image_original, mask, palette, pixelate_image, show_image, ski):
    lab_palette = ski.color.rgb2lab(palette)
    lab_image_original = ski.color.rgb2lab(image_original[:,:, :3])
    pixel_image = pixelate_image(lab_image_original, 8, mask, lab_palette)
    show_image(pixel_image, 'Pixelated Image', size=10)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
