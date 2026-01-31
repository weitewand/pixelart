from pathlib import Path
import polars as pl
import requests
import time

SCRYFALL_CARD_URL = "https://api.scryfall.com/cards"
SLEEP_TIME = 0.11 #sec

def image_path(
        set_code: str, 
        collector_number: str, 
        face: str, 
        *,
        image_dir: Path = Path("images_raw"),
    ) -> Path:
    filename = f"{set_code.lower()}_{collector_number}_{face}.png"
    return image_dir / filename

def fetch_image_uris(
        decklist: pl.DataFrame, 
        deck_name: str,
        *,
        uri_lists_dir: Path = Path("image_uri_lists"),
    ) -> pl.DataFrame | None:

    path_to_image_uri_list = uri_lists_dir / (deck_name + ".parquet")

    # Ckeck if already fetched data:
    if path_to_image_uri_list.exists():
        return pl.read_parquet(path_to_image_uri_list)

    image_rows = []
    for row in decklist.iter_rows(named=True):
        set_code = row["set"]
        collector_number = row["collector_number"]

        request_url = SCRYFALL_CARD_URL + '/' + set_code + '/' + collector_number
        r = requests.get(request_url, timeout=10)
        time.sleep(SLEEP_TIME)

        if r.status_code != 200:
            image_uri_front = "Request failed"
            image_uri_back = "Request failed"
            double_faced = None
            image_rows.append(
                {
                    **row,
                    "double_faced": double_faced,
                    "image_uri_front": image_uri_front,
                    "image_uri_back": image_uri_back,
                }
            )
            continue

        data = r.json()

        # Handle single-faced vs double-faced cards
        if "image_uris" in data:
            image_uri_front = data["image_uris"]["png"]
            image_uri_back = None
            double_faced = False
        elif "card_faces" in data:
            image_uri_front = data["card_faces"][0]["image_uris"]["png"]
            image_uri_back = data["card_faces"][1]["image_uris"]["png"]
            double_faced = True
            #frame = data["frame"] + "_" + data["frame_effects"][0]

        # Get information about the frame around the art:
        #if "frame_effects" in data:
            #if "legendary" in data["frame_effects"] and data['']:
                #frame_front = "_".join([data["frame"], data["frame_effects"]])

        image_rows.append(
            {
                **row,
                "double_faced": double_faced,
                "image_uri_front": image_uri_front,
                "image_uri_back": image_uri_back,
                #"frame_front": frame_front
            }
        )    

    image_uris = pl.DataFrame(image_rows)
    image_uris.write_parquet(path_to_image_uri_list)
    return image_uris

def fetch_and_save_card_image(row: list) -> list[Path] | None:
    set_code = row["set"]
    collector_number = row["collector_number"]
    path_front = image_path(set_code, collector_number, "front")
    path_back = image_path(set_code, collector_number, "back")

    # Do not re-download
    if path_back.exists():
        return [path_front, path_back]
    elif path_front.exists():
        return [path_front, None]

    img = requests.get(row["image_uri_front"], timeout = 10)
    time.sleep(SLEEP_TIME)
    img.raise_for_status()
    path_front.write_bytes(img.content)

    if row["double_faced"] == True:  
        img = requests.get(row["image_uri_back"], timeout=10)
        img.raise_for_status()
        path_back.write_bytes(img.content)
        return [path_front, path_back]

    return [path_front, None]

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size] 

def fetch_filenames(
    folder: Path,
    extensions: list[str] = None,
) -> list[str]:
    """
    Fetches all filenames from the specified `folder`

    Parameters
    ----------
    folder : Path
        Folder to fetch the filenames from, given as a 
        posix-path.
    extensions : list[str]
        Return only filenames that have a extension that is 
        in `extensions`. If None, return all filenames. Defaults
        to None.

    Return
    ------
    All filenames that match the `extensions` from the sepcified
    `folder`.
    """
    if extensions == None:
        return [p.name for p in folder.iterdir() if p.is_file()]

    extensions = [ext.lower() for ext in extensions]
    return [
        p.name 
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
