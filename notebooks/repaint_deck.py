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

    import importlib
    import os
    from pathlib import Path
    import marimo as mo
    import numpy as np
    from pathlib import Path
    import re
    import polars as pl
    import time
    import requests
    import skimage as ski
    import pixelart_lib.io as io
    import pixelart_lib.visualisation as vis
    import pixelart_lib.pixelise as pix

    importlib.reload(io)
    importlib.reload(vis)
    importlib.reload(pix)

    IMAGE_WIDTH = 150  # px
    IMAGES_PER_ROW = 5
    return Path, io, mo, np, os, pix, pl, ski, vis


@app.cell(hide_code=True)
def _(Path, mo, os):
    path_to_decklists = os.path.join(os.getcwd(), 'decklists')
    folder = Path(path_to_decklists)

    txt_files = sorted(p.name for p in folder.glob("*.txt"))

    decklist_select = mo.ui.radio(
        options=txt_files,
        value = "test.txt",
        label="Select a list to repaint",

    )

    decklist_select
    return decklist_select, path_to_decklists


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Careful: The parquet-file with the image urls has to be deleted if a decklist is changed without changing its name.
    """)
    return


@app.cell
def _(Path, decklist_select, io, mo, path_to_decklists, pl, ski):
    path_to_decklist = Path(path_to_decklists + "/" + decklist_select.value)
    lines = path_to_decklist.read_text().splitlines()
    regex_archidekt = r"^\s*(\d+)x\s+(.+?)\s+\(([a-zA-Z0-9]+)\)\s+([0-9A-Za-z-]+)\s+\[([^\]]+)\]\s*$"

    decklist_raw = (
        pl.DataFrame({"line": lines})
        .with_columns(
            pl.col("line").str.extract_groups(
                regex_archidekt        
            ).alias("g")
        )
        .unnest("g")
        .rename(
            {
                "1": "qty",
                "2": "name",
                "3": "set",
                "4": "collector_number",
                "5": "category",
            }
        )
        .with_columns(
            pl.col("qty").cast(pl.Int64),
            pl.col("set").str.to_uppercase(),
        )
        #.drop('line')
    )

    decklist = decklist_raw.drop('line')

    bad_lines = decklist_raw.filter(pl.col("qty").is_null()).drop(pl.all().exclude('line'))

    decklist_images = io.fetch_image_uris(decklist, decklist_select.value[0:-4])

    image_rows = []

    for row in decklist_images.iter_rows(named=True):
        paths = io.fetch_and_save_card_image(row)
        image_rows.append(
            {
                **row,
                "image_path_front": paths[0],
                "image_path_back": paths[1],
            }
        )

    df_images = pl.DataFrame(image_rows)


    images = [
        ski.io.imread(path)[:,:,:3]
        for row in df_images.iter_rows(named=True)
        for face, path in (
            ("front", row["image_path_front"]),
            ("back", row["image_path_back"]),
        )
        if path not in {None, "Request failed"}
    ]

    images_rendered = [mo.image(img, width=150) for img in images]

    def create_grid(
        elements,
        elements_per_row,
    ):
        grid = mo.vstack(
            [
                mo.hstack(row, justify="start")
                for row in io.chunked(elements, elements_per_row)
            ]
        )
        return grid

    grid_simple_images = create_grid(images_rendered, 4)

    images[0].shape
    return create_grid, image_rows, images


@app.cell
def _(image_rows, images, ski):
    image = ski.io.imread(image_rows[0]["image_path_front"])
    images[0].shape
    return


app._unparsable_cell(
    r"""

    num_images = len(images)  # images already defined somewhere

    # Fetch all existing masks
    available_masks = io.fetch_filenames(Path(\"masks\"), extensions=[\".npy\"])

    dropdowns = [
        mo.ui.dropdown(
            options=available_masks,
            value=\"2015_normal_1.npy\",
            #label=f\"Overlay {i}\",
        )
        for i in range(num_images)
    ]

    default_mask = np.load(Path(\"masks/2015_normal_1.npy\"))

    gallery_items = [
        mo.vstack(
            [
                mo.image(vis.overlay_mask(image, mask=default_mask, colour=(255,255,255), alpha=.8, frame=False), width=400),
                dd
            ], 
            align=\"center\"
        )
        for image, dd in zip(images, dropdowns)
    ]

    grid = create_grid(elements=gallery_items, elements_per_row=)

    grid
    """,
    name="_"
)


@app.cell
def _(available_masks, np):
    masks = {
        key: np.load(f"masks/{key}")
        for key in available_masks
    }
    return (masks,)


@app.cell
def _(create_grid, dropdowns, images, masks, mo, vis):
    cards_frames = [
        mo.vstack([ #leads to automatic scaling of width to fit view
            mo.image(
                vis.overlay_mask(
                    image, mask=masks[dd.value], 
                    colour=(255,255,255), 
                    alpha=.8, 
                    frame=False), 
                width=400
            )
            ])
        for image, dd in zip(images, dropdowns)
    ]

    grid_frames = create_grid(elements=cards_frames, elements_per_row=3)

    cards_and_frames = [
        [
            card,
            masks[dd.value],
        ]
        for i, card, dd in zip(list(range(len(images))), images, dropdowns)
    ]

    grid_frames
    return (cards_and_frames,)


@app.cell
def _(cards_and_frames, create_grid, mo, pix):
    pixelised_cards = [
        pix.pixelise_card(image=img, mask=mask)
        for img, mask in cards_and_frames
    ]

    gallery_pixelised = [
        mo.vstack([
            mo.image(card, width=400)
        ])
        for card in pixelised_cards
    ]

    create_grid(gallery_pixelised, elements_per_row=3)
    return (pixelised_cards,)


@app.cell
def _(mo):
    save_cards = mo.ui.button(label="Save cards")
    save_cards
    return


@app.cell
def _(pixelised_cards):
    pixelised_cards[33]
    return


@app.cell
def _(pixelised_cards):
    import imageio.v3 as iio

    iio.imwrite("test/Jeskas_Will.png", (pixelised_cards[34]*255).astype("uint8"))

    return


if __name__ == "__main__":
    app.run()
