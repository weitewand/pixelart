import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    # Project root = parent of the notebooks folder
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import importlib
    import marimo as mo
    import skimage as ski
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial import KDTree
    import pixelart_lib.visualisation as vis
    import pixelart_lib.pixelise as pix
    import pixelart_lib.basics as bsc

    def cut_out_image(image, mask):
        pass

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


    return bsc, mo, np, os, pix, show_palette, ski


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load image
    """)
    return


@app.cell
def _(mo, os, ski):
    path_to_images = os.path.join(os.getcwd(), 'images_raw', 'fdn_106_front.png')
    image_original = ski.io.imread(path_to_images)
    image_original = ski.util.img_as_float(image_original)
    image_rgb = ski.color.rgba2rgb(image_original)
    image_gray = ski.color.rgb2gray(image_rgb)
    [height, width] = image_gray.shape
    mo.image(image_rgb, width=300)
    #show_image(image_rgb, 'RGB Image')
    #show_image(image_gray, 'Gray Image')
    return image_original, image_rgb


@app.cell
def _(image_rgb, mo, np, os):
    path_to_mask = os.path.join(os.getcwd(), 'masks', '2015_normal_1.npy')
    mask = np.load(path_to_mask)
    mo.image(image_rgb * mask[..., None])
    print(mask.dtype, np.unique(mask))
    return (mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Preprocessing
    """)
    return


@app.cell
def _(bsc, image_rgb, mask, mo):
    art_only = bsc.extract_area((image_rgb*255).astype("uint8"), mask)
    mo.image(art_only, width=200)
    return (art_only,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pixelate it!
    """)
    return


@app.cell
def _(art_only, pix, show_palette):
    palette = pix.create_palette(art_only, 12)
    show_palette(palette)
    return (palette,)


@app.cell
def _(image_original, mask, mo, palette, pixelate_image, ski):
    lab_palette = ski.color.rgb2lab(palette)
    lab_image_original = ski.color.rgb2lab(image_original[:,:, :3])
    pixel_image = pixelate_image(lab_image_original, 8, mask, lab_palette)
    #show_image(pixel_image, 'Pixelated Image', size=10)
    mo.hstack([
        mo.image(image_original, width=250), 
        mo.image(pixel_image, width=250)
    ])
    return


@app.cell
def _(image_rgb, mask, mo, pix):
    pixel_card = pix.pixelise_card((image_rgb*255).astype("uint8"), mask, n_colours=16, size_px=8)
    mo.image(pixel_card, width=400)
    return


@app.cell
def _():
    return


@app.cell
def _(image_rgb, mask, mo, pix):
    colours = [4,8,16]
    sizes = [4,8,16]
    rows = []
    for n_c in colours:
        row = []
        for size in sizes:
            row.append(mo.image(
                pix.pixelise_card(
                    image=(image_rgb*255).astype("uint8"),
                    mask=mask,
                    n_colours=n_c,
                    size_px=size
                ),
                caption=f"n_c:{n_c}, size:{size}"
            ))
        rows.append(mo.hstack(row))
    rows
    return


if __name__ == "__main__":
    app.run()
