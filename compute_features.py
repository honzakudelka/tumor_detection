from feature_extraction import structural, textural
import cv2
import pandas as pd
import numpy as np
import sys
from timeit import default_timer as timer
from datetime import timedelta
import os
from microscopyio.slide_image import NDPISlideImage, CamelyonSlideImage
import random

from hp_utils import background_subtraction


def get_patch_metadata(input_file, output_file=None):
    meta_tags = ['loc_x', 'loc_y', 'sz_x', 'sz_y', 'label', 'orig_file']
    instances = {'TU': [], 'BO': [], 'NO': []}

    skip_rows = 0
    if output_file is not None and os.path.exists(output_file):
        with open(output_file) as csv_ofile:
            odf = pd.read_csv(csv_ofile, delimiter=';', index_col=False, skipinitialspace=True)

            skip_rows = odf.shape[0]

            if 'aug_' in output_file:
                skip_rows = skip_rows // 6

    with open(input_file) as csv_file:
        try:

            df = pd.read_csv(csv_file, delimiter=';', index_col=False,
                             skipinitialspace=True, skiprows=range(1, skip_rows + 1))

            col_names = df.columns.values
            col_indices = []
            for tag in meta_tags:
                col_indices += [idx for idx, name in enumerate(col_names)
                                if name == tag]

            for row in df.itertuples():
                row_arr = np.asarray([row[i + 1] for i in col_indices])

                # store patch information as ((loc_x, loc_y), label, file)
                instances[row_arr[4]].append(((int(row_arr[0]), int(row_arr[1])),
                                              (int(row_arr[2]), int(row_arr[3])),
                                              row_arr[4], row_arr[5]))

        except ValueError as e:
            print("Failed to parse file {0} \n Exception: \n ----------- \n {1}".format(
                input_file,
                str(e)
            ))
            return None

    return instances


def print_dictarr_to_csv(dictarr, o_file, append=False):
    CSV = ""
    key_line = ""
    if not append:
        for k, v in dictarr[0].items():
            key_line += "{};".format(k)

        CSV += key_line + "\n"

    for feature_dict in dictarr:
        line = ""
        for k, v in feature_dict.items():
            line += "{};".format(v)
        CSV += line + "\n"

    if not append:
        with open(o_file, 'w') as f:
            f.write(CSV)
    else:
        with open(o_file, 'a') as f:
            f.write(CSV)

    f.close()


def random_separation(im, patch_list, n_augm=5, is_camelyon=False, swap=True):
    # rgb_from_her = np.array([[0.4605, 0.7538, 0.3914],
    #                          [0.2948, 0.7491, 0.5873],
    #                          [0.2720, 0.8782, 0.3852]])
    rgb_from_her = np.array([
        [0.47680668, 0.54035088, 0.33853616],
        [0.33593786, 0.74302103, 0.58208704],
        [0.42269833, 0.80876244, 0.37791299]])

    in_modulation_matrix = np.array([[0.02, 0.14, 0.083],
                                     [0.06, 0.02, 0.05],
                                     [0.03, 0.05, 0.03]])

    # rgb_from_her = np.array([[0.4640, 0.8384, 0.2101],
    #                          [0.3083, 0.7571, 0.5725],
    #                          [0.2376, 0.8919, 0.3788]])
    if is_camelyon:
        rgb_from_her = np.array([[0.7595018, 0.51920101, 0.38165572],
                                 [0.4895436, 0.74380669, 0.50788103],
                                 [0.53518641, 0.76756465, 0.35352657]])

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

    limit = 1e-7
    if is_camelyon:
        limit = 1e-7

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
        # volHi = np.sum(2 * (im_H > 0.9).astype(np.float)) + 1e-3
        volE = np.sum(2 * (im_E > 0.1).astype(np.float))

        # print("H/E Scores: {} {}, VOL {} {} vE/vH {} vHi/vH {}".format(HE_diff,
        #                                                                volHi, volH, volE,
        #                                                                float(volE) / float(volH),
        #                                                                float(volHi) / float(volH)))

        score = np.abs(score_mean) * score_var * 10 * (
        1 + np.exp(-float(volE) / float(volH)))  # 1e3 * abs(HE_diff) / HE_div

        # print("[{}] [{}] raw:".format(n_run, score) + str(score_mean) + ", " + str(score_var))
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

            # r_stains = np.zeros_like(stains)
            # h_stains = np.zeros_like(stains)
            # e_stains = np.zeros_like(stains)
            #
            # im_sep = stains
            #
            # r_stains[:, 0] = np.copy(im_sep[:, 0])
            #
            # im_R = 255 * (np.exp(np.dot(-r_stains, a_rgb_from_her)))
            # imR = np.reshape(np.clip(im_R, 0, 255), im.shape).astype(np.uint8)
            #
            # h_stains[:, 1] = np.copy(im_sep[:, 1])
            # im_H = 255 * (1 - np.exp(np.dot(-h_stains, a_rgb_from_her)))
            # imH = np.reshape(np.clip(im_H, 0, 255), im.shape).astype(np.uint8)
            #
            # e_stains[:, 2] = np.copy(im_sep[:, 2])
            # im_E = 255 * (1 - np.exp(np.dot(-e_stains, a_rgb_from_her)))
            # imE = np.reshape(np.clip(im_E, 0, 255), im.shape).astype(np.uint8)
            #
            # w_str = "Channel Separation + Augmentation"
            # cv2.namedWindow(w_str, cv2.WINDOW_NORMAL)
            # cv2.resizeWindow(w_str, 1200, 300)
            # cv2.imshow(w_str, np.hstack([im.astype(np.uint8),
            #                              np.reshape(255*im_H, im.shape).astype(np.uint8),
            #                              np.reshape(255*im_E, im.shape).astype(np.uint8),
            #                              np.reshape(255*im_R, im.shape).astype(np.uint8),
            #                              ]))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

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
    im_H = 255 * (1 - np.exp(np.dot(-h_stains, best_mixing)))
    imH = np.reshape(np.clip(im_H, 0, 255), im.shape).astype(np.uint8)

    e_stains[:, 2] = np.copy(im_sep[:, 2])
    im_E = 255 * (1 - np.exp(np.dot(-e_stains, best_mixing)))
    imE = np.reshape(np.clip(im_E, 0, 255), im.shape).astype(np.uint8)

    # im_A = 255 * np.exp( np.dot(-stains, rgb_from_her))
    # if swap:
    #     im_A[:, [0, 2]] = im_A[:, [2, 0]]
    # imA = np.reshape(np.clip(im_A, 0, 255), im.shape).astype(np.uint8)


    # w_str = "Channel Separation + Augmentation"
    # cv2.namedWindow(w_str, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(w_str, 1200, 300)
    # cv2.imshow(w_str, np.hstack([im.astype(np.uint8),
    #                              imA,
    #                              imH.astype(np.uint8),
    #                              imE.astype(np.uint8),
    #                              imR.astype(np.uint8),
    #                              ]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return cv2.cvtColor(imH, cv2.COLOR_RGB2GRAY), \
           cv2.cvtColor(imE, cv2.COLOR_RGB2GRAY), \
           cv2.cvtColor(imR, cv2.COLOR_RGB2GRAY)


def separation(im, patch_list, n_augm=0, is_camelyon=False, swap=True):
    rgb_from_her = np.array([[0.4605, 0.7538, 0.3914],
                             [0.2948, 0.7491, 0.5873],
                             [0.2720, 0.8782, 0.3852]])

    # rgb_from_her = np.array([[0.4640, 0.8384, 0.2101],
    #                          [0.3083, 0.7571, 0.5725],
    #                          [0.2376, 0.8919, 0.3788]])
    if is_camelyon:
        rgb_from_her = np.array([[0.5298, 0.8073, 0.2544],
                                 [0.4258, 0.7706, 0.4715],
                                 [0.2188, 0.8858, 0.3651]])

    her_from_rgb = np.linalg.inv(rgb_from_her)

    rgb = 1 / 255.0 * (np.copy(im).astype(np.float32) + 1)
    rgb = rgb.reshape((-1, 3))
    if swap:
        rgb[:, [0, 2]] = rgb[:, [2, 0]]

    stains = np.dot(-np.log(rgb), her_from_rgb)

    r_stains = np.zeros_like(stains)
    h_stains = np.zeros_like(stains)
    e_stains = np.zeros_like(stains)

    im_sep = stains

    for i in range(n_augm):

        augm_alpha = 1.0 + 0.01 * np.random.randint(-10, 10, 3)
        augm_beta = 0.01 * np.random.randint(-10, 10, 3)

        augm_stains = augm_alpha * stains + augm_beta
        im_A = 255 * np.exp(np.dot(-augm_stains, rgb_from_her))
        if swap:
            im_A[:, [0, 2]] = im_A[:, [2, 0]]
        imA = np.reshape(np.clip(im_A, 0, 255), im.shape).astype(np.uint8)

        if n_augm == 1:
            im_sep = augm_stains

        patch_list.append(imA.astype(np.uint8))

    r_stains[:, 0] = np.copy(im_sep[:, 0])
    im_R = 255 * np.exp(np.dot(-r_stains, rgb_from_her))
    imR = cv2.cvtColor(np.reshape(np.clip(im_R, 0, 255), im.shape).astype(np.uint8),
                       cv2.COLOR_RGB2GRAY)

    h_stains[:, 1] = np.copy(im_sep[:, 1])
    im_H = 255 * (1 - np.exp(np.dot(-h_stains, rgb_from_her)))
    imH = np.reshape(np.clip(im_H, 0, 255), im.shape).astype(np.uint8)[:, :, 1]

    e_stains[:, 2] = np.copy(im_sep[:, 2])
    im_E = 255 * (1 - np.exp(np.dot(-e_stains, rgb_from_her)))
    imE = np.reshape(np.clip(im_E, 0, 255), im.shape).astype(np.uint8)[:, :, 1]

    return imH, imE, imR


if __name__ == "__main__":

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    max_no_patches = int(sys.argv[3])
    patch_extract_level = int(sys.argv[4])
    options = ['hist']
    if len(sys.argv) > 5:
        options = sys.argv[5:]

    patches_in = get_patch_metadata(input_file, output_file)

    if patches_in is None or (len(patches_in['NO']) + len(patches_in['TU']) + len(patches_in['BO'])) < 1:
        print("Nothing to do for input directory " + input_file)
        quit()

    print("Retrieved: {} Normal, {} Tumor and {} Border patches".format(len(patches_in['NO']),
                                                                        len(patches_in['TU']),
                                                                        len(patches_in['BO']))
          )

    patches = []
    # patches += patches_in['TU'] + patches_in['BO']

    if 'show' in options:
        random.shuffle(patches_in['TU'])
        patches += patches_in['TU'][:10]  # + patches_in['BO']
    else:
        patches += patches_in['TU'] + patches_in['BO']

    visited_no_patches = 0

    if 0 < max_no_patches < len(patches_in['NO']):
        random.shuffle(patches_in['NO'])
        patches += patches_in['NO'][:max_no_patches]
    else:
        patches += patches_in['NO']

    if len(patches) < 1:
        print("Nothing to do for input directory " + input_file)
        quit()

    image_features = []
    n = len(patches)
    i = 1
    avg_time = 0
    cum_time = 0

    dsigma = np.array([0.25, 0.125, 0.0625])
    show_intermediate = False
    if 'show' in options:
        show_intermediate = True

    if patch_extract_level > 1:
        dsigma *= 2 * patch_extract_level

    si = CamelyonSlideImage(patches[0][3], None)
    is_camelyon = True
    area_low = 100
    if patches[0][3].find('.ndpi') > -1:
        si = NDPISlideImage(patches[0][3], None)
        is_camelyon = False
        area_low = 70

    for patch_descriptor in patches:
        location = patch_descriptor[0]
        p_size = patch_descriptor[1]
        label = patch_descriptor[2]

        if 'doublesize' in options:
            p_size = (2 * p_size[0], 2 * p_size[1])

        print("[{}] {} @ {}".format(label, p_size, location))

        orig_patch = si.load_patch(location, p_size, patch_extract_level)
        # mean filtering to reduce mosaicking present in the data
        ori_patch = cv2.cvtColor(np.copy(orig_patch), cv2.COLOR_RGBA2RGB)
        ori_patch_mf = cv2.medianBlur(ori_patch, 3)

        if p_size[0] // pow(2, patch_extract_level) < 128:
            orig_patch = ori_patch_mf
        else:
            orig_patch = background_subtraction.subtract_background(ori_patch_mf, light_background=True, radius=20,
                                                                    down_factor=1)

        # orig_patch = ori_patch_mf
        tstart = timer()

        n_augm = 0
        probe_count = []
        if 'randsep' in options:
            uH, uE, uR = random_separation(np.copy(orig_patch), probe_count, n_augm=20, is_camelyon=is_camelyon)
        else:
            uH, uE, uR = separation(np.copy(orig_patch), probe_count, n_augm=n_augm, is_camelyon=is_camelyon)

        if n_augm < 1:
            probe_count = [orig_patch]

        res_energy = np.mean(uR) / 255.0
        uHEt = 255 * (uH > 0.7 * np.max(uH)).astype(np.uint8)
        uHEt = cv2.dilate(uHEt,
                          kernel=np.ones((3, 3), np.uint8),
                          iterations=1)

        augm_id = 0
        for aug_patch in probe_count:
            patch = cv2.cvtColor(aug_patch, cv2.COLOR_RGB2BGR)
            vis_patch = None
            if show_intermediate:
                vis_patch = np.copy(aug_patch)

            structural_features = {}
            p_subpatch_sz = (p_size[0] / 4, p_size[1] / 4)
            # if 'stru' in options:
            #     structural_features = structural.get_structural_features(patch, subpatch=p_subpatch_sz,
            #                                                              visualize=False, detect_nuclei=False)
            # elif 'strn' in options:
            #     structural_features = structural.get_structural_features(patch, subpatch=p_subpatch_sz,
            #                                                              visualize=False)

            bins = np.linspace(0, 255, 17)
            uH_hist, uH_bins = np.histogram(uH, bins, density=True)
            uE_hist, uE_bins = np.histogram(uE, bins, density=True)
            cB_hist, cB_bins = np.histogram(patch[:, :, 0], bins, density=True)
            cG_hist, cG_bins = np.histogram(patch[:, :, 1], bins, density=True)
            cR_hist, cR_bins = np.histogram(patch[:, :, 2], bins, density=True)

            for h_i in range(len(bins) - 1):
                structural_features.update({"hist_uE_bin_{:02d}".format(h_i): uE_hist[h_i]})
                structural_features.update({"hist_uH_bin_{:02d}".format(h_i): uH_hist[h_i]})
                structural_features.update({"hist_cB_bin_{:02d}".format(h_i): cB_hist[h_i]})
                structural_features.update({"hist_cG_bin_{:02d}".format(h_i): cG_hist[h_i]})
                structural_features.update({"hist_cR_bin_{:02d}".format(h_i): cR_hist[h_i]})

            Himg_thr_er_im, Himg_thr_kpoints = structural.blob_detection_watershed(uHEt,
                                                                                   pow(2, patch_extract_level),
                                                                                   None, area_low=area_low)

            structural_features.update(structural.he_structural_features(uH, uE, uR, (4, 4)))

            if show_intermediate:
                cv2.namedWindow("Channel Separation", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Channel Separation", 1400, 900)
                cv2.imshow("Channel Separation", np.hstack([orig_patch,
                                                            # cv2.cvtColor(Htr_nc_im.astype(np.uint8), cv2.COLOR_GRAY2RGB),
                                                            # cv2.cvtColor(uHE, cv2.COLOR_GRAY2RGB),
                                                            cv2.cvtColor(uH, cv2.COLOR_GRAY2RGB),
                                                            cv2.cvtColor(uE, cv2.COLOR_GRAY2RGB),
                                                            cv2.cvtColor(uR, cv2.COLOR_GRAY2RGB),
                                                            # cv2.cvtColor(uHEt, cv2.COLOR_GRAY2RGB),
                                                            # Himg_thr_er_im
                                                            ]))
                rv = cv2.waitKey(8000)
                cv2.destroyAllWindows()

                # cv2.imwrite("/tmp/patch_{}_{}_bc.tif".format(
                #     '_'.join(os.path.basename(input_file).split('_')[:2]),
                #     i),
                #     orig_patch
                # )

            subpatch_nuclei = list()
            for ii in range(4):
                for jj in range(4):
                    subpatch_nuclei.append(len(filter(
                        lambda x: ii * p_subpatch_sz[0] < x[0] < (ii + 1) * p_subpatch_sz[0] and jj * p_subpatch_sz[1] <
                                                                                                 x[1] < (jj + 1) *
                                                                                                        p_subpatch_sz[
                                                                                                            1],
                        Himg_thr_kpoints
                    )))

            if len(Himg_thr_kpoints) > 0:
                sp_nuclei = (16. * np.array(subpatch_nuclei, dtype=np.float)) / len(Himg_thr_kpoints)
            else:
                sp_nuclei = np.array(subpatch_nuclei, dtype=np.float)

            structural_features.update({"sp_num_nuclei_tot": len(Himg_thr_kpoints),
                                        "sp_num_nuclei_span": np.max(sp_nuclei) - np.min(sp_nuclei),
                                        "sp_num_nuclei_var": np.var(sp_nuclei)})

            textural_features = dict()  # textural.color_channel_histogram(patch)
            textural_features.update({"wx_residual_energy": res_energy})

            grey_norm = np.zeros_like(patch)
            grey_norm = cv2.normalize(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), dst=grey_norm,
                                      alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            if 'text' in options:
                uHnorm = uH.astype(np.float) / 255.0
                uEnorm = uE.astype(np.float) / 255.0
                uRnorm = uR.astype(np.float) / 255.0
                Gimg = grey_norm.astype(np.float) / 255.0

                textural_features.update(textural.get_wavelet_responses(Gimg, "cG_16", (4, 4)))
                textural_features.update(textural.get_wavelet_responses(uHnorm, "cH_16", (4, 4)))
                textural_features.update(textural.get_wavelet_responses(uEnorm, "cE_16", (4, 4)))
                # wavelet_res_features = textural.get_wavelet_responses(uRnorm, "cR_16", (4, 4))

                textural_features.update(textural.get_wavelet_responses(Gimg, "cG_4", (2, 2)))
                textural_features.update(textural.get_wavelet_responses(uHnorm, "cH_4", (2, 2)))
                textural_features.update(textural.get_wavelet_responses(uEnorm, "cE_4", (2, 2)))

                textural_features.update(textural.get_wavelet_responses(Gimg, "cG_1", (1, 1), maxlevel=6))
                textural_features.update(textural.get_wavelet_responses(uHnorm, "cH_1", (1, 1), maxlevel=6))
                textural_features.update(textural.get_wavelet_responses(uEnorm, "cE_1", (1, 1), maxlevel=6))
                textural_features.update(textural.get_wavelet_responses(uRnorm, "cR_1", (1, 1), maxlevel=6))

                textural_features.update(
                    textural.get_wavelet_responses(uHnorm, "cH_1ds1", (1, 1), maxlevel=6, downsample=dsigma[0]))
                textural_features.update(
                    textural.get_wavelet_responses(uHnorm, "cH_1ds2", (1, 1), maxlevel=6, downsample=dsigma[1]))
                textural_features.update(
                    textural.get_wavelet_responses(uHnorm, "cH_1ds3", (1, 1), maxlevel=6, downsample=dsigma[2]))

                textural_features.update(
                    textural.get_wavelet_responses(uEnorm, "cE_1ds1", (1, 1), maxlevel=6, downsample=dsigma[0]))
                textural_features.update(
                    textural.get_wavelet_responses(uEnorm, "cE_1ds2", (1, 1), maxlevel=6, downsample=dsigma[1]))
                textural_features.update(
                    textural.get_wavelet_responses(uEnorm, "cE_1ds3", (1, 1), maxlevel=6, downsample=dsigma[2]))

            tend = timer()

            merged_features = dict(textural_features, **structural_features)
            image_features.append(dict({"loc_x": location[0], "loc_y": location[1], "label": label, "he_augm": augm_id},
                                       **merged_features))
            augm_id += 1

        cum_time += tend - tstart
        avg_time = cum_time / (1. * i)

        eta = (n - i) * avg_time

        eta_str = timedelta(seconds=eta)
        print(" ...processed patch {0:03d}/{1} in {2:.3f} [s] \t eta = {3}".format(i, n, tend - tstart, eta_str))
        i += 1

        # Terminate job after elapsed cumulative time, it must be < 4h when spawn into the SGE's fastjobs queue
        if cum_time > 120 * 60:
            break

    print_dictarr_to_csv(image_features, output_file, append=os.path.exists(output_file))
