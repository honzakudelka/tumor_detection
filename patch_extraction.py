"""
    Extract patch meta-data from a microscopy image (given a tissue mask file and an annotation file, when available)

    Input is a microscopy image (currently tiff or ndpi) and a tissue mask file.

    Output is a CSV file containing the coordinates of the upper left corner for each valid
    patch (tissue coverage condition, etc.) A tissue mask must be provided, optionally also a tumor annotation
    file can be specified (in such case, the labels of each patch will be set accordingly).
    If the --preview option is specified, also a png file with visualized patch boundaries is written to the specified
    path.

    Author:
        Jan Hering (BIA/CMP)
        jan.hering@fel.cvut.cz
"""
from collections import OrderedDict
import argparse

import cv2

from microscopyio.slide_image import NDPISlideImage, CamelyonSlideImage
from hp_utils.dict_to_csv import dictlist_to_csv

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract patches from microscopy (histopat.) images")

    parser.add_argument("-i", "--image", type=str, help='Input image', default=None, required=True)
    parser.add_argument("-m", "--mask", type=str, help='Mask image, tissue class with max value within',
                        default=None, required=True)

    opt_group = parser.add_argument_group("Optional arguments")
    opt_group.add_argument("-l", "--extract-level", type=int, help='Level for extracting patches',
                           default=6, required=False)
    opt_group.add_argument("-p", "--patch-size", type=int, help='Size of the patch (on the base level 0)',
                           default=1024, required=False)
    opt_group.add_argument("-s", "--patch-stride", type=int, help='Offset to next patch (on the base level 0)',
                           default=-1, required=False)
    opt_group.add_argument("-t", "--tumor-annot", type=str,
                           help='Tumor annotation file (default: searching for standard locations/extensions)',
                           default=None, required=False)
    opt_group.add_argument("--min-coverage", type=float, help='Minimal tissue coverage in accepted patches',
                           default=0.95, required=False)
    opt_group.add_argument("--csv", type=str, help='CSV file for saving the complete patch metadata',
                           default=None, required=False)
    opt_group.add_argument("--preview", type=str, help='File path for storing the preview image with annotated patches',
                           default=None, required=False)

    cmdline_args = parser.parse_args()

    si = CamelyonSlideImage(image_path=cmdline_args.image,
                            tissue_mask_path=cmdline_args.mask, tumor_annotation_file=cmdline_args.tumor_annot)
    if '.ndpi' in cmdline_args.image:
        si = NDPISlideImage(image_path=cmdline_args.image,
                            tissue_mask_path=cmdline_args.mask, tumor_annotation_file=cmdline_args.tumor_annot)

    p_shift = cmdline_args.patch_stride
    if p_shift < 0:
        p_shift = cmdline_args.patch_size

    p = si.get_annotated_patches(extract_level=cmdline_args.extract_level,
                                 min_coverage_extraction=cmdline_args.min_coverage,
                                 min_tumor_coverage=0.6, p_size=cmdline_args.patch_size, p_shift=p_shift)

    if cmdline_args.preview is not None:
        p_img = si.get_patch_visualization(cmdline_args.extract_level, p, cmdline_args.patch_size, show=False)
        cv2.imwrite(cmdline_args.preview, p_img)

    if cmdline_args.csv is not None:
        patch_metadata = []

        for patch in p:
            location = patch[0]
            label = patch[1]
            patch_metadata.append(
                OrderedDict(
                    [("loc_x", location[0]), ("loc_y", location[1]),
                     ("sz_x", cmdline_args.patch_size),
                     ("sz_y", cmdline_args.patch_size),
                     ("label", label),
                     ("orig_file", cmdline_args.image), ("mask", cmdline_args.mask)
                     ]
                )
            )

        dictlist_to_csv(patch_metadata, cmdline_args.csv)

    # si._visualize = True
    # for patch in p:
    #     patch0 = si.load_patch(patch[0], cmdline_args.patch_size)

    quit(0)
