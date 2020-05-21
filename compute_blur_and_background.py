import argparse
import random

import cv2
import numpy as np

from microscopyio.patch_utils import get_patch_metadata
from microscopyio.slide_image import NDPISlideImage, CamelyonSlideImage

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute features for given input file")

    parser.add_argument("-i", "--input", type=str, help='Input CSV file with patch meta description',
                        default=None, required=True)
    parser.add_argument("-o", "--output", type=str, help='Output file to store computed features within',
                        default=None, required=True)

    opt_group = parser.add_argument_group("Optional arguments")
    opt_group.add_argument("-m", "--max-patches", type=int,
                           help='Set maximum number of (normal) patches (default -1 to extract all)',
                           default=-1, required=False)
    opt_group.add_argument("-t", "--time-limit", type=int,
                           help='Maximum running time in minutes (default 120)',
                           default=120, required=False)
    opt_group.add_argument("-l", "--extract_level", type=int,
                           help='Image level used for patch extraction',
                           default=0, required=False)
    opt_group.add_argument("-cm", "--convolution-matrix", type=str,
                           help='String representation of the color-mixing matrix, direction rgb to HEres',
                           default=None, required=False)
    opt_group.add_argument("-md", "--modulation-matrix", type=str,
                           help='Modulation matrix for separation optimization'
                                ' (add option \'randsep\' to the list \'-e\'',
                           default=None, required=False)

    cmdline_args = parser.parse_args()

    input_file = cmdline_args.input
    output_file = cmdline_args.output
    max_no_patches = cmdline_args.max_patches
    base_extract_level = cmdline_args.extract_level

    rgb_from_her = None
    her_from_rgb = None
    modulation_matrix = None
    if cmdline_args.convolution_matrix is not None:
        rgb_from_her = eval(cmdline_args.convolution_matrix)
        print('Retrieved color-mixing matrix: \n ' + str(rgb_from_her))

        her_from_rgb = np.linalg.inv(rgb_from_her)

    if cmdline_args.modulation_matrix is not None:
        modulation_matrix = eval(cmdline_args.modulation_matrix)
        print('Retrieved modulation matrix: \n ' + str(modulation_matrix))

    patches_in = get_patch_metadata(input_file)

    if patches_in is None or (len(patches_in['NO']) + len(patches_in['TU']) + len(patches_in['BO'])) < 1:
        print("Nothing to do for input directory " + input_file)
        quit()

    print("Retrieved: {} Normal, {} Tumor and {} Border patches".format(len(patches_in['NO']),
                                                                        len(patches_in['TU']),
                                                                        len(patches_in['BO']))
          )

    patches = []

    visited_no_patches = 0
    if 0 < max_no_patches < len(patches_in['NO']):
        random.shuffle(patches_in['NO'])
        patches += patches_in['NO'][:max_no_patches]
    else:
        patches += patches_in['NO']

    if len(patches) < 1:
        print("Nothing to do for input directory " + input_file)
        quit()

    n = len(patches)
    i = 1

    si = CamelyonSlideImage(patches[0][3], None)
    is_camelyon = True
    area_low = 100
    if patches[0][3].find('.ndpi') > -1:
        si = NDPISlideImage(patches[0][3], None)
        is_camelyon = False
        area_low = 50

    blur_score = []
    bckg_score = []

    patch_vis_meta = []
    tot_area = 1. / ((patches[0][1])[0] * (patches[0][1])[1])

    for patch_descriptor in patches:
        p_location = patch_descriptor[0]
        p_base_size = patch_descriptor[1]
        label = patch_descriptor[2]

        p_size = (p_base_size[0], p_base_size[1])


        location = (
            p_location[0] - (p_size[0] - p_base_size[0]) // 2, p_location[1] - (p_size[1] - p_base_size[1]) // 2)

        orig_patch = si.load_patch(location, p_size, base_extract_level)
        ori_patch = cv2.cvtColor(np.copy(orig_patch), cv2.COLOR_RGBA2RGB)
        #   ori_patch_mf = cv2.medianBlur(ori_patch, 3)

        gr_patch = cv2.cvtColor(ori_patch, cv2.COLOR_RGB2GRAY)

        blurriness = cv2.Laplacian(gr_patch, cv2.CV_8U).var()
        background = np.sum((gr_patch > 200).astype(np.uint8)) * tot_area
        blur_score.append(blurriness)
        bckg_score.append(background)

        # if blurriness > 2:
        #     w_name = "Blurriness {:.2f} | Bckg ratio {:.2f}".format(blurriness, background)
        #     cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow(w_name, 900, 900)
        #     cv2.imshow(w_name, gr_patch)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        patch_vis_meta.append([p_location, label])

    output_map = []
    lower_clip = 20
    upper_clip = 200 * (3**base_extract_level)
    np_blur = np.asarray(blur_score)
    np_blur = np.clip(np_blur, lower_clip, upper_clip) - lower_clip
    np_blur *= -100. / upper_clip
    np_blur += 100.

    np_bckg = 100. * np.asarray(bckg_score)

    patchvis_blur = si.get_patch_visualization(display_level=6, patch_list=patch_vis_meta, patch_size=p_size[0],
                                               scalars=np_blur, border_scalars=None,
                                               line_thickness=-1,
                                               show=False, filled=True, output_map=output_map)

    if output_map is not None and len(output_map):
        cv2.imwrite(output_file+'_blur.png', output_map[0])

    output_map = []
    patchvis_bckg = si.get_patch_visualization(display_level=6, patch_list=patch_vis_meta, patch_size=p_size[0],
                                               scalars=np_bckg, border_scalars=None,
                                               line_thickness=-1,
                                               show=False, filled=True, output_map=output_map)

    if output_map is not None and len(output_map):
        cv2.imwrite(output_file+'_bckg.png', output_map[0])

