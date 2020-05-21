from PIL import Image

import numpy as np

from microscopyio import hp_tiff as tiff

if __name__ == "__main__":
    # CAMELYON'16
    annon_root_16 = "/datagrid/personal/herinjan/data_camelyon16/Train-Ground_Truth/XML"
    raw_img, annot_img, amask = tiff.read_image("/local/herinjan/Downloads/tumor_026.tif",
                                                annon_root=annon_root_16, level=8)

    # annot_img.show()
    # amask.save("/local/temporary/histopato/tumor_095_l7.tif")

    # estimate boundaries of dense tissue areas
    cvraw_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2GRAY)
    # cv2.imwrite("/local/temporary/histopato/tumor_095_raw_l7.tif", cvraw_img)

    cvraw_blur = cv2.GaussianBlur(cvraw_img, (7, 7), 3)
    cvret, cvraw_thr = cv2.threshold(cvraw_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cvraw_thr_inv = cv2.bitwise_not(cvraw_thr)

    # cv_fill = cvraw_thr.copy()
    # kernel = np.ones((5, 5), np.uint8)
    # cv_open = cv2.morphologyEx(cv_fill, cv2.MORPH_OPEN, kernel)
    #
    # pil_ff = Image.fromarray(cv_open)
    # pil_ff.show()

    sample_mask = cvraw_thr_inv + amask

    ## Overlay estimated tissue boundary with the annotation mask
    pil_bgmask = Image.fromarray(sample_mask)
    # pil_bgmask.show()
    # out = Image.blend(pil_bgmask, amask.convert('L'), 0.4)
    pil_bgmask.show()

    # cv2.imwrite("/local/temporary/histopato/tumor_095_op_l7.tif", cvraw_thr)
