import os
import sys

import cv2
import numpy as np
from openslide import OpenSlide

if __name__ == "__main__":

    # PETACC
    extract_level = 6
    visualize = False

    image_file = sys.argv[1]
    image_type = sys.argv[2]
    image_name = os.path.splitext(os.path.basename(image_file))[0]

    slide_img = OpenSlide(image_file)
    img = slide_img.read_region((0, 0), extract_level, size=slide_img.level_dimensions[extract_level])
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    output_image = sys.argv[3] + "MASK_" + image_type + "_" + image_name + "_l{}.tif".format(extract_level)

    if os.path.exists(output_image):
        print("\t file exists...")
        quit()

    print("Output: {}".format(output_image))
    cv2.imwrite(output_image, image)

    quit(1)
