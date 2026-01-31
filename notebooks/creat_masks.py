import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    # Project root = parent of the notebooks folder
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import skimage as ski
    import os
    import numpy as np
    from sklearn.cluster import KMeans
    from scipy.spatial import KDTree
    import pixelart_lib.visualisation as vis
    import marimo as mo
    import altair as alt
    import polars as pl
    import importlib

    importlib.reload(vis)

    def cut_out_image(image, mask):
        pass
    return Path, mo, np, os, ski, vis


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load Image
    """)
    return


@app.cell
def _(mo, os, ski):
    path_to_images = os.path.join(os.getcwd(), 'images_raw', 'one_175_front.png')
    image_original = ski.io.imread(path_to_images)
    image_rgb = image_original[:, :, :3]
    #image_rgb = ski.color.rgba2rgb(image_original)
    image_gray = ski.color.rgb2gray(image_rgb)
    [height, width] = image_gray.shape
    mo.image(image_original, width=800)
    return (image_gray,)


@app.cell
def _(image_gray, mo, ski, vis):
    # Assume image_gray is your [H, W] grayscale image with values in [0,1]

    # --- First, the image ---
    img_cell = mo.image(image_gray, width=200)

    # --- Then, the histogram ---
    hist_cell = vis.fig_histogram(image_gray, width=260)

    # --- Proposed threshold ---
    threshold_proposed = ski.filters.threshold_otsu(image_gray)

    # --- Display ---
    img_hist = mo.hstack([img_cell, hist_cell], widths="equal")
    slider_threshold = mo.ui.slider(
        start=0,
        stop=1,
        step=.01,
        value=threshold_proposed,
        orientation="horizontal",
        show_value=True,
    )
    mo.vstack([
        img_hist,
        mo.md(f"###Proposed threshold: {threshold_proposed}"), 
        slider_threshold,
    ])
    return (slider_threshold,)


@app.cell
def _(image_gray, mo, ski, slider_threshold, vis):
    binary_image = image_gray > slider_threshold.value

    seed = (200,200)
    filled_art = ski.segmentation.flood(binary_image, seed)
    filled_art = ski.morphology.remove_small_holes(filled_art, area_threshold=100000)

    mo.hstack([
        mo.image(vis.add_pixel_mark(binary_image.astype("uint8"), seed=seed), width=250),
        mo.image(filled_art.astype("uint8"), width=250)
    ])
    return (filled_art,)


@app.cell
def _(filled_art, mo):
    mask = filled_art.astype(bool)
    filename = mo.ui.text('Filename')
    save_mask = mo.ui.button(label="Save mask")
    mo.hstack([
        filename,
        save_mask,
    ])
    return filename, mask, save_mask


@app.cell
def _(Path, filename, mask, np, save_mask, ski):
    save_mask
    out_dir = Path("masks")
    output = out_dir / (filename.value + ".npy")
    if not output.exists():
        np.save(output, mask)
        ski.io.imsave(f"masks/{filename.value}.png", mask.astype("uint8")*255)
        print("Mask successfully saved")
    else:
        print("Mask not saved, it already existed")
    return


if __name__ == "__main__":
    app.run()
