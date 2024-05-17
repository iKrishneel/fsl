#!/usr/bin/env python

import colorsys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import cv2 as cv
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from igniter.logger import logger
from fsl.structures import Instances
from fsl.utils import colormap

_Image = Type[Image.Image]


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """
        Args:
            img: same as in __init__
        """
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


@dataclass
class Visualizer(object):
    image: Union[np.ndarray, _Image]
    scale: Optional[float] = 1.0
    font_scale: Optional[float] = 1.0
    area_thresh: Optional[int] = 1000

    def __post_init__(self):
        image = np.asarray(self.image)
        image = image * 255 if image.dtype == np.float32 else image
        image = image.clip(0, 255).astype(np.uint8)

        self.output = VisImage(image, scale=self.scale)
        self._default_font_size = (
            max(np.sqrt(self.output.height * self.output.width) // 90, 10 // self.scale) * self.font_scale
        )

    def __call__(self, instances: Instances, alpha: Optional[float] = 0.5, **kwargs: Dict[str, Any]) -> np.ndarray:
        return self.overlay(instances, alpha=alpha)

    def overlay(self, instances: Instances, colors: List[List[int]] = None, alpha: Optional[float] = 0.5):
        if len(instances) == 0:
            logger.warning('Nothing detected!')
            return self.output

        colors = colors or [colormap.random_color(rgb=True, maximum=1) for _ in range(len(instances))]
        instances = instances.convert_bbox_fmt('xywh').numpy().sort_by_area()

        for i in range(len(instances)):
            color = colors[i]
            if instances.bboxes is not None:
                self.draw_box(instances.bboxes[i], edge_color=color)
                text_pos = instances.bboxes[i]
                h_align = 'left'

            if instances.masks is not None:
                mask = instances.masks[i].astype(np.uint8) * 255
                contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                index = np.argmax([cv.contourArea(contour) for contour in contours])
                segment = contours[index].squeeze()
                self.draw_polygon(segment, color, alpha=alpha)
                text_pos = [*np.min(segment, axis=0).tolist(), *np.max(segment, axis=0).tolist()]
                h_align = 'center'

            if len(instances.labels):
                area = text_pos[2] * text_pos[3]
                if area < self.area_thresh * self.output.scale or text_pos[-1] < 40 * self.output.scale:
                    if text_pos[-1] + text_pos[1] >= self.output.height - 5:
                        text_pos = (text_pos[2] + text_pos[0], text_pos[1])
                    else:
                        text_pos = text_pos[:2]
                else:
                    text_pos = text_pos[:2]

                height_ratio = text_pos[-1] / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * self._default_font_size
                self.draw_text(
                    f'{instances.labels[i]}: {instances.scores[i]:.2f}',
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=h_align,
                    font_size=font_size,
                )

        return self.output

    def draw_box(
        self, box_coord: np.ndarray, alpha: float = 0.5, edge_color: str = 'g', line_style: str = '-'
    ) -> VisImage:
        x, y, width, height = box_coord.astype(np.intp)
        linewidth = max(self._default_font_size / 4, 1)
        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def draw_polygon(
        self, segment: np.ndarray, color: Tuple[float, ...], edge_color: Tuple[float, ...] = None, alpha: float = 0.5
    ) -> VisImage:
        if len(segment) == 0:
            return self.output

        if edge_color is None:
            # make edge color darker than the polygon color
            edge_color = self._change_color_brightness(color, brightness_factor=-0.7) if alpha > 0.8 else color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        self.output.ax.add_patch(polygon)
        return self.output

    def draw_text(
        self,
        text: str,
        position: Tuple[int, int],
        *,
        font_size: float = None,
        color: str = 'g',
        horizontal_alignment: str = 'center',
        rotation: int = 0,
    ) -> VisImage:
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="DejaVu Sans",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "green"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def _change_color_brightness(self, color: Tuple[float, ...], brightness_factor: float):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return tuple(np.clip(modified_color, 0.0, 1.0))


if __name__ == '__main__':
    bboxes = np.array(
        [
            [660, 258, 728, 335],
            [594, 159, 644, 205],
            [361, 373, 429, 446],
            [506, 45, 551, 91],
            [504, 240, 557, 282],
            [244, 340, 295, 386],
            [585, 471, 631, 523],
            [321, 257, 376, 298],
            [477, 133, 522, 181],
            [399, 219, 451, 257],
            [343, 445, 401, 487],
            [445, 294, 500, 331],
            [366, 129, 422, 171],
            [155, 227, 211, 265],
            [761, 342, 811, 422],
            [257, 104, 302, 155],
            [773, 193, 810, 246],
            [166, 427, 220, 462],
            [500, 378, 552, 450],
            [607, 456, 811, 574],
            [607, 288, 638, 392],
            [638, 399, 750, 529],
            [607, 399, 813, 574],
            [613, 0, 1022, 136],
            [638, 400, 813, 530],
            [343, 373, 429, 487],
        ],
        dtype=np.float32,
    )

    image = Image.open('/root/krishneel/Downloads/real_veggie_test_images/image_12.png')
    instances = Instances(
        image_height=574, image_width=1022, bboxes=bboxes, labels=['bean'] * len(bboxes), bbox_fmt='xyxy'
    )
    image = image.resize((instances.image_width, instances.image_height))

    v = Visualizer(image)
    x = v(instances)

    import matplotlib.pyplot as plt

    plt.imshow(x.get_image())
    # plt.imshow(x.img)
    plt.show()
