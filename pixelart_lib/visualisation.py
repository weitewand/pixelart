import marimo as mo
import numpy as np
import polars as pl
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from skimage.morphology import erosion, footprint_rectangle
from PIL import Image
from skimage.transform import resize

def resize_image(img, w, h):
    if is_binary_image(img):
        return (resize(
            img,
            (h, w),
            order=0 if img.dtype == np.bool_ else 1,
            preserve_range=True,
            anti_aliasing=False,
        )).astype(np.uint8)
    else:
        return (resize(
            img,
            (h, w),
            order=0 if img.dtype == np.bool_ else 1,
            preserve_range=True,
            anti_aliasing=True,
        )).astype(np.uint8)

def is_binary_image(img: np.ndarray) -> bool:
    if img.ndim != 2:
        return False

    values = np.unique(img)

    if values.size > 2:
        return False

    return set(values.tolist()).issubset({0, 1, 255, False, True})

def add_pixel_mark(
    image: np.ndarray,
    seed: tuple,
    *,
    colour: tuple = (255, 0, 0), # red
    size: int = 40,
    width: int = 4,
) -> np.ndarray:
    """
    Add a cross to the picture that marks the `seed`.

    Parameters
    ----------
    image : np.ndarray
        image to add the mark to.
    seed : List[int]
        coordinates for the mark.
    colour: tuple
        Colour of the mark in rgb, defaults to red.
    size : int
        Size of the mark in pixels, defaults to 20 px.
    width : int
        Width of the lines that make the mark in pixels, defaults to 2 px.

    Returns
    -------
    np.ndarray
        The image with the mark added
    
    Notes
    -----
    -   A `width` of 4 gives the same cross as a `width` of 5 (since 4 // 2 = 5 // 2). 
        Same is true for size.
    """
    img = image.copy()
    H, W = image.shape[:2]
    row = seed[0]
    column = seed[1]
    half_size = size // 2
    half_width = width // 2

    # Convert to RGB if grayscale or binary:
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.max() == 1:
        img = img*255

    # vertical distance from the seed pixwl:
    for d in range(-half_size, half_size + 1):
        # horizontal distance from the diagonal:
        for w in range(-half_width, half_width + 1):
            # main diagonal
            r1, c1 = row + d, column + d + w
            # anti diagonal
            r2, c2 = row - d, column + d + w
            # set colour only when pixel is in the image:
            if 0 <= r1 < H and 0 <= c1 < W:
                img[r1, c1] = colour
            if 0 <= r2 < H and 0 <= c2 < W:
                img[r2, c2] = colour

    return img

def fig_histogram(
    image: np.ndarray,
    *,
    bins: int = 256,
    log_scale: bool = True,
    width: int = 300,
    height: int = 200,
    title: str | None = None,
) -> mo.ui.altair_chart:
    """
    Create a marimo-native histogram using Altair for a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image with values in [0, 1].
    bina=s : int
        Number of histogram bins.
    log_scale : biil
        Weather to have a log scale for the y-axis.
    width, height : int
        Size of the chart.
    title : str | None
        Title of the chart, optional.
    
    Returns
    -------
    mo.ui.altair_chart
        A marimo UI element that can be stacked and layouted.
    """
    counts, bins = np.histogram(image, bins, range=(0, 1))
    df = pl.DataFrame({
        "bin_center": (bins[:-1] + bins[1:]) / 2,
        "count": counts,
    })

    y_scale = alt.Scale(type="log") if log_scale else alt.Scale()

    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            "bin_center:Q",
    #        bin=alt.Bin(maxbins=256),
            title="Pixel value",
            scale=alt.Scale(domain=[0, 1]),
        ),
        y=alt.Y(
            "count:Q", 
            title="Count",
            stack=None,
            scale=y_scale,
        ),
    ).properties(
        width=width,
        height=height,
    )

    if title is not None:
        chart = chart.properties(title=title)

    # Wrap the altair chart figure fo>r marimo display
    return mo.ui.altair_chart(hist)

def show_image(    
    image: np.ndarray,
    *,
    seed: tuple[int,int] | None = None,
    mask: np.ndarray | None = None,
    mask_mode: str = "border", # "overlay | "border"
    mask_alpha: float = 0.4,
    mask_colour: tuple = (255,0,0),
    width: int | None = None,
    height: int | None = None,
    title: str | None = None
):
    """
    Display an image in Marimo using plotly with optional overlays.

    The function supports **binary, grayscale, and RGB images** and allows
    adding a red cross at a given pixel coordinate (`seed`) or overlaying a 
    mask. This can be either done as a `overlay` or by oinly showing the 
    `border` of the amskd. The resulting chart is returned
    as a `mo.ui.plotly` that can be combined with other UI elements
    in Marimo.

    Parameters
    ----------
    image : np.ndarray
        The input image. Can be:
        - 2D array: binary or grayscale image with values in [0, 1]
        - 3D array: RGB image with shape (H, W, 3), values in [0, 1]
    seed : tuple[int, int], optional
        A pixel coordinate `(row, col)` where a red cross should be drawn.
        Useful for marking a reference point, seed of a flood, or key pixel.
        If None, no cross is drawn. Default is None.
    mask : np.ndarray, optional
        A array of the same pixel-size as `image` only contianing 0s and 1s. 
        The 1s mark the area that is shaded by the mask. If None, no mask is 
        overlayed.
        Useful for showing which area was determined to contain the art.
    mask_mode : str, optional
        A flag to denote wehater the amsk is `overlay`ed or just the `border`
        of the mask is drawn on top of `image`. Default is border.
    mask_alpha : float, optional
        A float specifing the opacity of the overlayed mask. Default is 0.4.
    mask_colour : tuple, optional
        A 3-tuple specifing the rgb-values of the mask. default is (255, 0, 0).
    width : int, optional
        Total width of the displayed chart in pixels. Height is then calculated 
        to preserve the ratio. When None, the size of `image` is used. Default
        is None.
    height : int, optional
        Total height of the displayed chart in pixels. Width is then calculated 
        to preserve the ratio. When None, the size of `image` is used. Default
        is None.
    title : str, optional
        Optional title for the chart. If None, no title is displayed.

    Returns
    -------
    mo.ui.plotly
        A Marimo Plotly chart displaying the image with optional overlays.
        Can be combined with `mo.hstack`, `mo.vstack`, or other Marimo UI elements.

    Notes
    -----
    - Overlays are rendered using `.add_trace()` with `go.scatter` and (cross and 
        border of the mask) or `.add_layout_image()` (overlayed mask).
    - Values in the image are expected to be normalized to `[0, 1]`. RGB images are automatically converted.
    - When both `width` and `height` are set, width takes precedence.
    """
    
    SIZE_X=40 #px
    WIDTH_X=2 #px

    H, W = image.shape[:2]

    # set width and height of display
    if width:
        display_width = width
        display_height = int(width * H / W)
        size_set = True
    elif height:
        display_height = height
        display_width = int(height * W / H)
        size_set = True
    else:
        size_set = False
   
    if size_set:
        img_disp = resize_image(image, display_width, display_height)
    else:
        img_disp = image.copy()

    # ---- base image ----
    if img_disp.ndim == 2:
        img_disp = np.stack([img_disp]*3, axis=-1)

    if np.issubdtype(img_disp.dtype, np.floating):
        img_disp = (np.clip(img_disp, 0, 1) * 255).astype(np.uint8)

    H, W = img_disp.shape[:2]

    _fig = go.Figure(go.Image(z=img_disp))

    # ---- mark seed pixel ----
    if seed is not None:
        _fig.add_trace(
            go.Scatter(
                y=[seed[0] - SIZE_X/2, seed[0] + SIZE_X/2], 
                x=[seed[1] - SIZE_X/2, seed[1] + SIZE_X/2],
                mode="lines",
                line={"color": "red", "width": WIDTH_X},
            )
        )
        _fig.add_trace(
            go.Scatter(
                y=[seed[0] - SIZE_X/2, seed[0] + SIZE_X/2],
                x=[seed[1] + SIZE_X/2, seed[1] - SIZE_X/2],
                mode="lines",
                line={"color": "red", "width": WIDTH_X},
            )
        )

    # ---- mask handling ----
    if mask is not None:
        mask = mask.astype(bool)

        if mask_mode == "overlay":
            rgba = mask_to_rgba(mask, mask_color, mask_alpha)
            pil = Image.fromarray(rgba, "RGBA")

            _fig.add_layout_image(
                source=pil,
                x=0,
                y=0,
                sizex=W,
                sizey=H,
                xref="x",
                yref="y",
                sizing="stretch",
                layer="above",
            )

        elif mask_mode == "border":
            boundary = mask_inner_boundary(mask)
            ys, xs = np.where(boundary)

            _fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="red", width=2),
                    showlegend=False,
                )
            )

        else:
            raise ValueError("mask_mode must be 'overlay' or 'border'")

    # ---- general config of plot ----
    layout_kwargs = dict(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
    )

    if title:
        layout_kwargs["title"]=dict(
            text=title, 
            x=0.5, 
            xanchor="center"
        )
    

    
    if size_set:
        layout_kwargs["width"] = display_width
        layout_kwargs["height"] = display_height

    _fig.update_layout(**layout_kwargs)

    _fig.update_xaxes(visible=False)
    _fig.update_yaxes(
        visible=False,
        autorange="reversed",
        scaleanchor="x",
    )

    return mo.ui.plotly(_fig)
    
def mask_to_rgba(mask, color=(255, 0, 0), alpha=0.4):
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = (mask.astype(np.uint8) * int(alpha * 255))
    return rgba

def mask_inner_boundary(mask):
    eroded = erosion(mask, square(3))
    return mask & (~eroded)

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    colour: tuple = (255, 0, 0),
    thickness: int = 5,
    frame: bool = True,
    alpha: float = .35,
) -> np.ndarray:
    """
    Overlays the given image with the mask or the edges of the mask.

    Parameters
    ----------
    image : np.ndarray
        rgb-Image to be overlayed.
    mask : np.ndarray
        Boolean image of same size as `image` that gives the area that is 
        masked. False-pixels are masked and True-pixels are not.
    colour : tuple
        Colour of the frame or shading that overlayed. Defaults to red.
    thickness : int
        Thickness in pixels of the line if `frame` == True. Defaults to 5.
    frame : bool
        Weather to overlay the a frame at the border of the mask ir to shade
        the whole masked area.
    alpha : float
        Translucency of the shade of the masked area. Defaults to 0.35.
    """
    img = image.copy()
    if frame == True:
        k = 2 * thickness + 1
        border = mask ^ erosion(~mask, footprint_rectangle((k, k)))
        img[border] = colour
        return img
    elif frame == False:
        mask = ~mask
        img_f = img.astype(np.float32)
        colour = np.array(colour, dtype=np.float32)
        img_f[mask] = (
            (1-alpha) * img_f[mask] + alpha * colour
        )
        return img_f.astype(np.uint8)
