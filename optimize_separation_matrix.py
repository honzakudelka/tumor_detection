import cv2
import numpy as np

import os
from microscopyio.slide_image import NDPISlideImage, CamelyonSlideImage
import argparse

from hp_utils import background_subtraction


def random_separation(im, patch_list, n_augm=5, is_camelyon=False, swap=True,
                      rgb_from_her=None, in_modulation_matrix=None, show=False):
    if rgb_from_her is None:
        rgb_from_her = np.array([
            [0.47680668, 0.54035088, 0.33853616],
            [0.33593786, 0.74302103, 0.58208704],
            [0.42269833, 0.80876244, 0.37791299]])
        if is_camelyon:
            rgb_from_her = np.array([[0.7595018, 0.51920101, 0.38165572],
                                     [0.4895436, 0.74380669, 0.50788103],
                                     [0.53518641, 0.76756465, 0.35352657]])

    if in_modulation_matrix is None:
        in_modulation_matrix = np.array([[0.02, 0.14, 0.083],
                                         [0.06, 0.02, 0.05],
                                         [0.03, 0.05, 0.03]])

        if is_camelyon:
            in_modulation_matrix = np.array([[0.08, 0.1, 0.14],
                                             [0.05, 0.02, 0.05],
                                             [0.02, 0.02, 0.02]])

    try_variations = True
    min_res = 10
    best_mixing = rgb_from_her
    best_mixing_inv = None
    n_run = 0
    min_runs = 3
    max_runs = n_augm

    rgb = 1 / 255.0 * (np.copy(im).astype(np.float32) + 1)
    rgb = rgb.reshape((-1, 3))
    if swap:
        rgb[:, [0, 2]] = rgb[:, [2, 0]]

    limit = 1e-5
    if is_camelyon:
        limit = 1e-5

    modulation_matrix = in_modulation_matrix

    while try_variations:
        a_rgb_from_her = rgb_from_her + np.multiply(modulation_matrix, (2 * np.random.rand(*rgb_from_her.shape) - 1))
        # print(a_rgb_from_her)
        # print("\n---\n")
        # print(a_rgb_from_her - rgb_from_her)
        a_her_from_rgb = np.linalg.inv(a_rgb_from_her)

        stains = np.dot(-np.log(rgb), a_her_from_rgb)

        r_stains = np.zeros_like(stains)
        h_stains = np.zeros_like(stains)
        e_stains = np.zeros_like(stains)

        r_stains[:, 0] = stains[:, 0]
        h_stains[:, 1] = stains[:, 1]
        e_stains[:, 2] = stains[:, 2]

        im_R = 1 - np.exp(np.dot(-r_stains, a_rgb_from_her))
        im_R = np.clip(im_R, 0, 1)
        score_var = np.var(im_R)
        score_mean = np.mean(im_R)

        im_H = 1 - np.exp(np.dot(-h_stains, a_rgb_from_her))
        im_H = np.clip(im_H, 0, 1)

        im_E = 1 - np.exp(np.dot(-e_stains, a_rgb_from_her))
        im_E = np.clip(im_E, 0, 1)

        HE_diff = np.mean(im_E - im_H)
        # HE_div = np.mean(im_E / (im_H+1e-4))

        volH = np.sum(2 * (im_H > 0.1).astype(np.float)) + 1e-3
        volHi = np.sum(2 * (im_H > 0.9).astype(np.float)) + 1e-3
        volE = np.sum(2 * (im_E > 0.1).astype(np.float))

        score = np.abs(score_mean) * score_var * 10 * (
        1 + np.exp(-float(volE) / float(volH)))  # 1e3 * abs(HE_diff) / HE_div

        if show:
            print("H/E Scores: {} VOL {} {} vE/vH {} vHi/vH {}".format(HE_diff,
                                                                       volH, volE,
                                                                       float(volE) / float(volH),
                                                                       float(volHi) / float(volH)))

            print("[{}] [{}] raw:".format(n_run, score) + str(score_mean) + ", " + str(score_var))
        if n_run < max_runs and (n_run < min_runs or not (score < limit)):

            if HE_diff < 0.1:
                modulation_matrix = np.copy(in_modulation_matrix)
                n_run += 1
                continue

            improvement = False

            if min_res > score:
                min_res = score
                best_mixing_inv = a_her_from_rgb
                best_mixing = a_rgb_from_her
                id_best = n_run
                improvement = True

            if improvement and score < limit * 1e2:
                rgb_from_her = np.copy(0.5 * (rgb_from_her + a_rgb_from_her))
                modulation_matrix = 0.75 * modulation_matrix
                # print("Adapt:" + str(modulation_matrix[0]))

            if n_run > max_runs / 2 and min_res < limit * 1e2:
                try_variations = False
                continue

            try_variations = True

            n_run += 1
            continue
        else:

            if min_res > score:
                min_res = score
                best_mixing_inv = a_her_from_rgb
                best_mixing = a_rgb_from_her
                id_best = n_run

            try_variations = False

    stains = np.dot(-np.log(rgb), best_mixing_inv)

    r_stains = np.zeros_like(stains)
    h_stains = np.zeros_like(stains)
    e_stains = np.zeros_like(stains)

    im_sep = stains

    r_stains[:, 0] = np.copy(im_sep[:, 0])

    im_R = 255 * (np.exp(np.dot(-r_stains, best_mixing)))
    imR = np.reshape(np.clip(im_R, 0, 255), im.shape).astype(np.uint8)

    h_stains[:, 1] = np.copy(im_sep[:, 1])
    im_H = 255 * (np.exp(np.dot(-h_stains, best_mixing)))
    imH = np.reshape(np.clip(im_H, 0, 255), im.shape).astype(np.uint8)

    e_stains[:, 2] = np.copy(im_sep[:, 2])
    im_E = 255 * (np.exp(np.dot(-e_stains, best_mixing)))
    imE = np.reshape(np.clip(im_E, 0, 255), im.shape).astype(np.uint8)

    if show:
        w_str = "Channel Separation + Augmentation"
        cv2.namedWindow(w_str, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(w_str, 1200, 300)
        cv2.imshow(w_str, np.hstack([im.astype(np.uint8),
                                     np.reshape(imH, im.shape).astype(np.uint8),
                                     np.reshape(imE, im.shape).astype(np.uint8),
                                     np.reshape(imR, im.shape).astype(np.uint8),
                                     ]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return best_mixing


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute optimized channel separation matrix")

    parser.add_argument("-i", "--image", type=str, help='Input image', default=None, required=True)
    parser.add_argument("-m", "--mask", type=str, help='Mask image, tissue class with max value within',
                        default=None, required=True)
    parser.add_argument("-o", "--csv-out", type=str, help="CSV File to store the median estimated matrix")

    opt_group = parser.add_argument_group("Optional arguments")
    opt_group.add_argument("-l", "--extract-level", type=int, help='Level for extracting patches',
                           default=6, required=False)
    opt_group.add_argument("-p", "--patch-size", type=int, help='Size of the patch (on the base level 0)',
                           default=1024, required=False)
    opt_group.add_argument("--min-coverage", type=float, help='Minimal tissue coverage in accepted patches',
                           default=0.95, required=False)
    opt_group.add_argument("--n_samples", type=int, help='Number of extracted samples',
                           default=8, required=False)
    opt_group.add_argument("-cm", "--convolution-matrix", type=str,
                           help='String representation of the color-mixing matrix, direction rgb to HEres',
                           default=None, required=False)
    opt_group.add_argument("-md", "--modulation-matrix", type=str,
                           help='Modulation matrix for separation optimization'
                                ' (add option \'randsep\' to the list \'-e\'',
                           default=None, required=False)
    opt_group.add_argument("--preview", help='Flag whether to show intermediate results',
                           action='store_true')

    cmdline_args = parser.parse_args()

    si = None
    if '.ndpi' not in cmdline_args.image:
        si = CamelyonSlideImage(image_path=cmdline_args.image,
                                tissue_mask_path=cmdline_args.mask)
        is_camelyon = True
        rgb_from_her = np.array([[0.7595018, 0.51920101, 0.38165572],
                                 [0.4895436, 0.74380669, 0.50788103],
                                 [0.53518641, 0.76756465, 0.35352657]])
        modulation_matrix = None

    if cmdline_args.convolution_matrix is not None:
        rgb_from_her = eval(cmdline_args.convolution_matrix)
        print('Retrieved color-mixing matrix: \n ' + str(rgb_from_her))

        her_from_rgb = np.linalg.inv(rgb_from_her)

    if cmdline_args.modulation_matrix is not None:
        modulation_matrix = eval(cmdline_args.modulation_matrix)
        print('Retrieved modulation matrix: \n ' + str(modulation_matrix))

    if '.ndpi' in cmdline_args.image:
        si = NDPISlideImage(image_path=cmdline_args.image,
                            tissue_mask_path=cmdline_args.mask)
        is_camelyon = False

        rgb_from_her = np.array([
            [0.47680668, 0.54035088, 0.33853616],
            [0.33593786, 0.74302103, 0.58208704],
            [0.42269833, 0.80876244, 0.37791299]])

    p_shift = cmdline_args.patch_size
    p = si.get_annotated_patches(extract_level=cmdline_args.extract_level,
                                 min_coverage_extraction=cmdline_args.min_coverage,
                                 min_tumor_coverage=0.6, p_size=cmdline_args.patch_size, p_shift=p_shift)

    np.random.shuffle(p)
    sample_patches = p[:cmdline_args.n_samples]

    separation_matrix = []
    for sid, sample in enumerate(sample_patches):
        img = si.load_patch(sample[0], (cmdline_args.patch_size, cmdline_args.patch_size), 0)
        image = np.array(img)

        ori_patch_mf = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB), 3)
        orig_patch_bc = background_subtraction.subtract_background(ori_patch_mf, light_background=True, radius=20,
                                                                   down_factor=1)

        init_deconv = rgb_from_her
        # Pay more credit to the improving estimate, move initialization near the common estimated matrix...
        #   ... should speed-up the optimization a bit
        if len(separation_matrix) > 0:
            init_deconv = np.median(np.array(separation_matrix + [rgb_from_her]), axis=0)

        Mdeconv = random_separation(np.copy(orig_patch_bc), None, n_augm=20, is_camelyon=is_camelyon,
                                    rgb_from_her=init_deconv, in_modulation_matrix=modulation_matrix,
                                    show=cmdline_args.preview)

        separation_matrix.append(Mdeconv)

    M_median = np.median(np.array(separation_matrix), axis=0)

    with open(cmdline_args.csv_out, 'w') as fout:
        fout.write("{};np.{}\n".format(os.path.basename(cmdline_args.image),
                                       repr(M_median).replace('\n', '')
                                       )
                   )

    fout.close()
