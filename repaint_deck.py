import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


app._unparsable_cell(
    r"""
    import os
    from pathlib import Path
    import marimo as mo
    from pathlib import Path
    import re
    import polars as pl
    import time
    import requests

    IMAGE_DIR = Path(\"images_raw\")
    IMAGE_DIR.mkdir(exist_ok=True)
    IMAGE_URI_LISTS_DIR = Path(\"image_uri_lists\")
    IMAGE_URI_LISTS_DIR.mkdir(exist_ok=True)
    SLEEP_TIME = 0.11 #sec
    SCRYFALL_CARD_URL = \"https://api.scryfall.com/cards\"
    IMAGE_WIDTH = 150  # px
    IMAGES_PER_ROW = 5

    def image_path(set_code: str, collector_number: str, face: str) -> Path:
        filename = f\"{set_code.lower()}_{collector_number}_{face}.png\"
        return IMAGE_DIR / filename

    def fetch_image_urls(decklist: pl.DataFrame, deck_name: str) -> pl.DataFrame | None:
        path_to_image_uri_list = IMAGE_URI_LISTS_DIR / (deck_name + \".parquet\")

        # Ckeck if already fetched data:
        if path_to_image_uri_list.exists():
            return pl.read_parquet(path_to_image_uri_list)
    
        image_rows = []
        for row in decklist.iter_rows(named=True):
            set_code = row[\"set\"]
            collector_number = row[\"collector_number\"]
        
            request_url = SCRYFALL_CARD_URL + '/' + set_code + '/' + collector_number
            r = requests.get(request_url, timeout=10)
            time.sleep(SLEEP_TIME)

            if r.status_code != 200:
                image_uri_front = \"Request failed\"
                image_uri_back = \"Request failed\"
                double_faced = None
                image_rows.append(
                    {
                        **row,
                        \"double_faced\": double_faced,
                        \"image_uri_front\": image_uri_front,
                        \"image_uri_back\": image_uri_back,
                    }
                )
                continue
        
            data = r.json()
    
            # Handle single-faced vs double-faced cards
            if \"image_uris\" in data:
                image_uri_front = data[\"image_uris\"][\"png\"]
                image_uri_back = None
                double_faced = False
            elif \"card_faces\" in data:
                image_uri_front = data[\"card_faces\"][0][\"image_uris\"][\"png\"]
                image_uri_back = data[\"card_faces\"][1][\"image_uris\"][\"png\"]
                double_faced = True
                frame = data[\"frame\"] + \"_\" + data[\"frame_effects\"][0]

            # Get information about the frame around the art:
            if \"frame_effects\" in data:
                if \"legendary\" in data[\"frame_effects\"] and data['']:
                    frame_front = \"_\".join([data[\"frame\"], data[\"frame_effects\"]]
        
            image_rows.append(
                {
                    **row,
                    \"double_faced\": double_faced,
                    \"image_uri_front\": image_uri_front,
                    \"image_uri_back\": image_uri_back,
                    \"frame_front\": frame_front
                }
            )    

        image_uris = pl.DataFrame(image_rows)
        image_uris.write_parquet(path_to_image_uri_list)
        return image_uris
    
    def fetch_and_save_card_image(row: list) -> list[Path] | None:
        set_code = row[\"set\"]
        collector_number = row[\"collector_number\"]
        path_front = image_path(set_code, collector_number, \"front\")
        path_back = image_path(set_code, collector_number, \"back\")

        # ✅ Do not re-download
        if path_back.exists():
            return [path_front, path_back]
        elif path_front.exists():
            return [path_front, None]

        img = requests.get(row[\"image_uri_front\"], timeout = 10)
        time.sleep(SLEEP_TIME)
        img.raise_for_status()
        path_front.write_bytes(img.content)

        if row[\"double_faced\"] == True:  
            img = requests.get(row[\"image_uri_back\"], timeout=10)
            img.raise_for_status()
            path_back.write_bytes(img.content)
            return [path_front, path_back]
    
        return [path_front, None]

    def chunked(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i : i + size] 
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(Path, mo, os):
    path_to_decklists = os.path.join(os.getcwd(), 'decklists')
    folder = Path(path_to_decklists)

    txt_files = sorted(p.name for p in folder.glob("*.txt"))

    decklist_select = mo.ui.radio(
        options=txt_files,
        value = "kiblerhugs.txt",
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
def _(
    IMAGES_PER_ROW,
    IMAGE_WIDTH,
    Path,
    chunked,
    decklist_select,
    fetch_and_save_card_image,
    fetch_image_urls,
    mo,
    path_to_decklists,
    pl,
):
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

    decklist_images = fetch_image_urls(decklist, decklist_select.value[0:-4])

    image_rows = []

    for row in decklist_images.iter_rows(named=True):
        paths = fetch_and_save_card_image(row)
        image_rows.append(
            {
                **row,
                "image_path_front": paths[0],
                "image_path_back": paths[1],
            }
        )

    df_images = pl.DataFrame(image_rows)

    images = [
        mo.image(
            path,
            #caption=f'{row["name"]} ({face})',
            width=IMAGE_WIDTH,
        )
        for row in df_images.iter_rows(named=True)
        for face, path in (
            ("front", row["image_path_front"]),
            ("back", row["image_path_back"]),
        )
        if path not in {None, "Request failed"}
    ]


    grid = mo.vstack(
        [
            mo.hstack(row, justify="start")
            for row in chunked(images, IMAGES_PER_ROW)
        ]
    )

    df_images
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
