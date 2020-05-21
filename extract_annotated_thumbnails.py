from microscopyio.slide_image import NDPISlideImage, CamelyonSlideImage
from hp_utils.dict_to_csv import dictlist_to_csv
from collections import OrderedDict
import cv2
import argparse
import numpy as np
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract ")

    parser.add_argument("-i", "--image", type=str, help='Input image', default=None, required=True)
    parser.add_argument("-o", "--output_prefix", type=str, help='Output image prefix', default=None, required=True)
    parser.add_argument("-m", "--mask", type=str, help='Mask image, tissue class with max value within',
                        default=None, required=True)
    parser.add_argument("-ml", "--mask-level", type=int, help='Extraction level of the mask image',
                        default=6, required=True)
    parser.add_argument("-t", "--tumor-annot", type=str,
                           help='Tumor annotation file (default: searching for standard locations/extensions)',
                           default=None, required=False)

    opt_group = parser.add_argument_group("Optional arguments")
    opt_group.add_argument("-l", "--extract-level", type=int, help='Level for extracting patches',
                           default=3, required=False)

    cmdline_args = parser.parse_args()

    si = CamelyonSlideImage(image_path=cmdline_args.image,
                            tissue_mask_path=cmdline_args.mask, tumor_annotation_file=cmdline_args.tumor_annot)

    tissue_mask_img = cv2.imread(cmdline_args.mask, cv2.IMREAD_GRAYSCALE)
    tissue_mask_img -= np.min(tissue_mask_img)

    im2, contours, hierarchy = cv2.findContours(tissue_mask_img,
                                                mode=cv2.RETR_CCOMP,
                                                method=cv2.CHAIN_APPROX_NONE)

    for ci, c in enumerate(contours):

        minx = im2.size
        miny = im2.size

        maxx = 0
        maxy = 0

        if len(c) < 100:
            continue

        c_min_x = np.min(c[:, 0, 0])
        c_max_x = np.max(c[:, 0, 0])
        c_min_y = np.min(c[:, 0, 1])
        c_max_y = np.max(c[:, 0, 1])

        minx = min(minx, c_min_x)
        maxx = max(maxx, c_max_x)
        miny = min(miny, c_min_y)
        maxy = max(maxy, c_max_y)

        scale_factor = pow(2, cmdline_args.mask_level-cmdline_args.extract_level)
        extract_location = (minx * scale_factor, miny * scale_factor)

        if cmdline_args.output_prefix[:-1] == "show@":

            level = int(cmdline_args.output_prefix.split('@')[1])
            show_scale_factor = pow(2, cmdline_args.mask_level)
            show_extract_location = (minx * show_scale_factor, miny * show_scale_factor)
            size_scale_factor = pow(2, cmdline_args.mask_level - level)
            show_extract_size = ((maxx-minx)*show_scale_factor, (maxy-miny)*show_scale_factor)
            #show_extract_size = (2048, 2048)

            show_patch = si.load_patch(show_extract_location, show_extract_size, level)

            w_name = "Image {}, contour {}".format(os.path.basename(cmdline_args.image), ci)
            cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(w_name, 900, 900)
            cv2.imshow(w_name, show_patch)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:

            tissue_img_outscale = np.zeros(((maxy - miny) * scale_factor, (maxx - minx) * scale_factor), dtype=np.uint8)

            cv2.resize(tissue_mask_img[miny:maxy, minx:maxx],
                       dst=tissue_img_outscale, dsize=None, fx=scale_factor, fy=scale_factor )

            tm = si._get_tumor_mask(cmdline_args.extract_level)
            tm_cut = tm[miny * scale_factor:maxy * scale_factor, minx * scale_factor:maxx * scale_factor]

            slide_image = si._np_img[miny * scale_factor:maxy * scale_factor, minx * scale_factor:maxx * scale_factor]

            cv2.imwrite(cmdline_args.output_prefix+"_lev{}_tissue_mask.png".format(cmdline_args.extract_level),
                        tissue_img_outscale)
            cv2.imwrite(cmdline_args.output_prefix + "_lev{}_tumor_mask.png".format(cmdline_args.extract_level),
                        tm_cut)
            cv2.imwrite(cmdline_args.output_prefix + "_lev{}_image.png".format(cmdline_args.extract_level),
                        slide_image)







