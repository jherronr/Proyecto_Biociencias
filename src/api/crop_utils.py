# src/api/crop_utils.py

import numpy as np
from PIL import Image

class CropBlackBorders:
    """
    Recorta bordes completamente negros de una imagen PIL (RGB).
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        img_gray = img.convert("L")
        np_img = np.array(img_gray)
        non_zero_rows = np.where(np_img.max(axis=1) > 0)[0]
        non_zero_cols = np.where(np_img.max(axis=0) > 0)[0]

        if non_zero_rows.size == 0 or non_zero_cols.size == 0:
            return img

        top_row, bottom_row = non_zero_rows[0], non_zero_rows[-1]
        left_col, right_col = non_zero_cols[0], non_zero_cols[-1]
        return img.crop((left_col, top_row, right_col + 1, bottom_row + 1))
