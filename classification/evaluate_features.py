import os
import argparse
from glob import glob

import numpy as np
import cv2

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

from skimage.filters import threshold_adaptive, rank
from skimage.morphology import disk

from microscopyio.slide_image import CamelyonSlideImage
from hp_utils import background_subtraction
from compute_features_context import random_separation, separation
from feature_extraction import textural, structural
#
# def get_features(patch_list):
#
#
# def visualize_texture_features(patch_list):
#

if __name__=="__main__":

    parser = argparse.ArgumentParser("Evaluate features for specific patches")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input directory ")
    parser.add_argument("-ip", "--input_prefix", type=str, required=True,
                        help="Prefix for selecting multiple subdirectories")
    parser.add_argument("-o", "--outfig", type=str, required=False, help="Path to output figure file")
    parser.add_argument("-l", "--level", type=int, required=False, default=0,
                        help="Patch extraction level")
    parser.add_argument("-sl", "--separation", type=int, default=0,
                        help="Indicate whether the wavelet features should be separated")

    args = parser.parse_args()
    base_extract_level = args.level

    image_features = []
    sorted_keys = None

    separate_wavelets = args.separation > 0
    ifile_root = args.input
    idir_list = sorted(glob(os.path.join(ifile_root, args.input_prefix)))

    for idir in idir_list:

        ifile = os.path.join(idir, 'exported_patch_list.csv')
        if not os.path.exists(ifile):
            print("File: {} not found".format(ifile))
            continue

        print("Computing patch features for [{}]".format(ifile))

        patches = []
        with open(ifile) as fpatches:

            for line in fpatches.readlines()[-20:]:
                if line[:5] == "file;":
                    continue

                data = line.split(';')
                patches.append( (data[0],
                                 (int(data[3]), int(data[4])),
                                 (int(data[5]), int(data[5])),
                                 data[2],
                                 float(data[-1])
                                 )
                                )

        if len(patches) < 1:
            raise RuntimeError("No patches retrieved.")

        fpath = ''
        mfile = '/datagrid/Medical/microscopy/CAMELYON16/patches/patches_meta_'
        if 'tumor' in patches[0][0]:
            fpath = '/datagrid/Medical/microscopy/CAMELYON16/training/tumor/{}.tif'.format(patches[0][0])
            mfile += 'tumor/tumor_mixing_matrix.txt'
        elif 'Normal' in patches[0][0]:
            fpath = '/datagrid/Medical/microscopy/CAMELYON16/training/normal/{}.tif'.format(patches[0][0])
            mfile += 'normal/normal_mixing_matrix.txt'
        else:
            fpath = '/datagrid/Medical/microscopy/CAMELYON16/testing/{}.tif'.format(patches[0][0])
            mfile += 'testing/testing_mixing_matrix.txt'

        if not os.path.exists(fpath):
            raise RuntimeError("Requested WSI file {} not found".format(fpath))

        # get matrix
        rgb_from_her = None
        her_from_rgb = None
        with open(mfile, 'r') as matf:
            for line in matf.readlines():
                lsplit = line.split(';')
                if lsplit[0] == '{}.tif'.format(patches[0][0]) and lsplit[1] is not 'None':
                    rgb_from_her = eval(lsplit[1])
                    break

        if rgb_from_her is not None:
            her_from_rgb = np.linalg.inv(rgb_from_her)

        show_intermediate = False
        level_prefix = 'L0'
        area_low = 100

        st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        st_elem_r1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

        si = CamelyonSlideImage(fpath, None)
        for patch_descriptor in patches:

            p_location = patch_descriptor[1]
            p_size = patch_descriptor[2]
            label = patch_descriptor[3]

            structural_features = {}
            textural_features = {}

            orig_patch = si.load_patch(p_location, p_size, base_extract_level)
            ori_patch = cv2.cvtColor(np.copy(orig_patch), cv2.COLOR_RGBA2RGB)
            ori_patch_mf = cv2.medianBlur(ori_patch, 3)

            if p_size[0] // pow(2, base_extract_level) < 256:
                orig_patch_in = ori_patch_mf
            else:
                orig_patch_in = background_subtraction.subtract_background(ori_patch_mf, light_background=True,
                                                                           radius=20, down_factor=1)

            orig_patch = np.copy(orig_patch_in)
            # uH, uE, uR = random_separation(np.copy(orig_patch), [], n_augm=20, is_camelyon=True,
            #                                rgb_from_her=rgb_from_her, in_modulation_matrix=None,
            #                                show=show_intermediate)
            uH, uE, uR = separation(np.copy(orig_patch), [], n_augm=1, is_camelyon=True,
                                    rgb_from_her=rgb_from_her, her_from_rgb=her_from_rgb)

            uH = cv2.GaussianBlur(uH, (0, 0), 1.0)
            uE = cv2.GaussianBlur(uE, (0, 0), 1.0)

            uE_mask = (uE > 10).astype(np.uint8)
            uH_mask = (uH < 0.65 * np.max(uH)).astype(np.uint8)
            uHE_mask = np.multiply(uE_mask, uH_mask)

            gr_patch = cv2.cvtColor(orig_patch, cv2.COLOR_RGB2GRAY)
            gr_patch_E = np.multiply(uHE_mask, gr_patch)

            local_otsu = rank.otsu(cv2.GaussianBlur(gr_patch_E, (0, 0), sigmaX=2.0), disk(15))
            binary_adaptive = 255 * (gr_patch_E < local_otsu).astype(np.uint8)

            uH_mask = (uH > 0.2 * np.max(uH)).astype(np.uint8)
            uE_val = np.multiply(uE, uH_mask)

            uHEt = 255 * (uH > 0.65 * np.max(uH)).astype(np.uint8)
            uHEt = cv2.morphologyEx(uHEt, cv2.MORPH_DILATE,
                                    kernel=st_elem_r1,
                                    iterations=1)

            uHEtmask = np.stack([uHEt < 250, uHEt < 250, uHEt < 250], axis=-1)
            for_hist = np.ma.array(orig_patch, copy=True, mask=uHEtmask)
            fh = for_hist.reshape((orig_patch.shape[0] * orig_patch.shape[1], 3))

            uHEt2 = np.multiply(cv2.morphologyEx(uHE_mask, cv2.MORPH_ERODE,
                                                 kernel=st_elem, iterations=2),
                                binary_adaptive
                                )

            vis_patch = None
            if show_intermediate:
                vis_patch = np.copy(ori_patch_mf)

            p_subpatch_sz = (p_size[0] / 4, p_size[1] / 4)

            # bins = np.linspace(0, 255, 17)
            # uH_hist, uH_bins = np.histogram(uH, bins, density=True)
            # uE_hist, uE_bins = np.histogram(uE, bins, density=True)
            #
            # for h_i in range(len(bins) - 1):
            #     structural_features.update({"L{}_hist_uE_bin_{:02d}".format(level_prefix, h_i): uE_hist[h_i]})
            #     structural_features.update({"L{}_hist_uH_bin_{:02d}".format(level_prefix, h_i): uH_hist[h_i]})

            Himg_thr_er_im, Himg_thr_kpoints = structural.blob_detection_watershed(uHEt,
                                                                                   pow(2, base_extract_level),
                                                                                   vis_patch, area_low=area_low,
                                                                                   skip_watershed=False)

            structural_features.update(structural.he_structural_features(uH, uE, uR,
                                                                         prefix="L{}_".format(level_prefix),
                                                                         subdivision=(4, 4)))

            Himg_thr_er_im2, Himg_thr_kpoints2 = structural.blob_detection_watershed(uHEt2,
                                                                                     pow(2, base_extract_level),
                                                                                     vis_patch,
                                                                                     area_low=2 * area_low,
                                                                                     area_high=1500,
                                                                                     skip_watershed=True)

            num_nuc_scale = 400.0/(ori_patch.shape[0]**2)

            if show_intermediate:
                wname = "Features {} |{}|{}|".format(patch_descriptor[0],
                                                     patch_descriptor[4],
                                                     patch_descriptor[3])
                cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(wname, 1400, 900)
                cv2.imshow(wname, np.vstack([np.hstack([cv2.cvtColor(uHEt, cv2.COLOR_GRAY2RGB),
                                                                       np.array((255 * Himg_thr_er_im),
                                                                                dtype=np.uint8),
                                                                       orig_patch,
                                                                       cv2.cvtColor(uE, cv2.COLOR_GRAY2RGB)

                                                                       ]),
                                                            np.hstack([
                                                                cv2.cvtColor(uHEt2, cv2.COLOR_GRAY2RGB),
                                                                np.array((255 * Himg_thr_er_im2),
                                                                         dtype=np.uint8),
                                                                cv2.cvtColor(uH, cv2.COLOR_GRAY2RGB),
                                                                cv2.cvtColor(binary_adaptive,
                                                                             cv2.COLOR_GRAY2RGB)
                                                            ])])
                           )
                rv = cv2.waitKey()
                cv2.destroyAllWindows()

            sp_n = 4
            if ori_patch.shape[0] < 512:
                sp_n = 2

            spatch_factorx = ori_patch.shape[0] // sp_n
            spatch_factory = ori_patch.shape[1] // sp_n

            subpatch_nuclei = list()
            for ii in range(sp_n):
                for jj in range(sp_n):
                    subpatch_nuclei.append(len(filter(
                        lambda x: ii * spatch_factorx < x[0] < (ii + 1) * spatch_factorx and
                                  jj * spatch_factory < x[1] < (jj + 1) * spatch_factory,
                        Himg_thr_kpoints
                    )))

            subpatch_nuclei2 = list()
            for ii in range(sp_n):
                for jj in range(sp_n):
                    subpatch_nuclei2.append(len(filter(
                        lambda x: ii * spatch_factorx < x[0] < (ii + 1) * spatch_factorx and
                                  jj * spatch_factory < x[1] < (jj + 1) * spatch_factory,
                        Himg_thr_kpoints2
                    )))

            sp_nuclei = num_nuc_scale * np.array(subpatch_nuclei, dtype=np.float)
            sp_nuclei2 = num_nuc_scale * np.array(subpatch_nuclei2, dtype=np.float)

            structural_features.update({"L{}_sp_num_nuclei_tot".format(level_prefix): num_nuc_scale * len(Himg_thr_kpoints),
                                        "L{}_sp_num_nuclei_span".format(level_prefix): np.max(
                                            sp_nuclei) - np.min(sp_nuclei),
                                        "L{}_sp_num_nuclei_var".format(level_prefix): np.var(sp_nuclei),
                                        "L{}_sp_num_diff_mean".format(level_prefix): np.mean(
                                            sp_nuclei - sp_nuclei2),
                                        "L{}_sp_num_diff_var".format(level_prefix): np.var(
                                            sp_nuclei - sp_nuclei2),
                                        "L{}_sp_num_largenuclei_tot".format(level_prefix): num_nuc_scale * len(
                                            Himg_thr_kpoints2),
                                        "L{}_sp_num_largenuclei_var".format(level_prefix): np.var(sp_nuclei2)
                                        })

            # textural_features.update({"L{}_wx_residual_energy".format(level_prefix): res_energy})

            uHnorm = uH.astype(np.float) / 255.0
            uEnorm = uE.astype(np.float) / 255.0
            uHEtnorm = uHEt.astype(np.float) / 255.0

            structural_features.update(
                textural.get_wavelet_responses(uHnorm, "L{}_cH_1".format(level_prefix),
                                               (1, 1), maxlevel=6, separate=separate_wavelets) )
            structural_features.update(
                textural.get_wavelet_responses(uEnorm, "L{}_cE_1".format(level_prefix),
                                               (1, 1), maxlevel=6, separate=separate_wavelets))
            structural_features.update(
                textural.get_wavelet_responses(uHEtnorm, "L{}_cHEt_1".format(level_prefix),
                                               (1, 1), maxlevel=6, separate=separate_wavelets))

            if sorted_keys is None:
                sorted_keys = sorted(structural_features.keys())

            values = []
            for key in sorted_keys:
                values.append( structural_features[key])

            lval = -1
            if label == 'TU':
                lval = 1

            image_features.append([lval, patch_descriptor[4]] + values)

    fp_feat = []
    fn_feat = []
    tp_feat = []
    tn_feat = []

    print('Features \n ------ \n ' + '\n'.join(sorted_keys))

    for feat in image_features:
        if feat[0] < 1:
            if feat[1] < 0.5:
                tn_feat.append(feat[2:])
            else:
                fp_feat.append(feat[2:])
        else:
            if feat[1] > 0.5:
                tp_feat.append(feat[2:])
            else:
                fn_feat.append(feat[2:])

    fp_feat = np.asarray(fp_feat)
    fn_feat = np.asarray(fn_feat)
    tp_feat = np.asarray(tp_feat)
    tn_feat = np.asarray(tn_feat)

    # for xi in range(len(fp_feat[0])):
    #
    fig1, ax_f1 = plt.subplots(1, 2, sharey=True)
    x = np.arange(1.0, tp_feat.shape[1], 1.0, dtype=float)

    bp_fp = ax_f1[0].boxplot(fp_feat, positions=x-0.15, widths=0.3, patch_artist=True)
    bp_tn = ax_f1[0].boxplot(tn_feat, positions=x+0.15, widths=0.3, patch_artist=True)

    for box in bp_fp['boxes']:
        box.set(facecolor='red')

    for box in bp_tn['boxes']:
        box.set(facecolor='green')

    ax_f1[0].set_title('Negative class (red=FP, green=TN')
    ax_f1[0].set_xticks(x)
    #ax_f1[0].set_yscale(nonposy='clip')
    ax_f1[0].grid(color='m', which='major', axis='y', ls='-', lw=1)

    bp_fn = ax_f1[1].boxplot(fn_feat, positions=x-0.15, widths=0.3, patch_artist=True)
    bp_tp = ax_f1[1].boxplot(tp_feat, positions=x+0.15, widths=0.3, patch_artist=True)

    for box in bp_fn['boxes']:
        box.set(facecolor='red')

    for box in bp_tp['boxes']:
        box.set(facecolor='green')

    ax_f1[1].set_title('Positive class (red=FN, green=TP)')
    #ax_f1[1].set_yscale(nonposy='clip')
    ax_f1[1].set_xticks(x)
    ax_f1[1].grid(color='m', which='major', axis='y', ls='-', lw=1)


    # ax[1][0].boxplot(fn_feat)
    # ax[1][0].set_title('FN')
    # ax[1][0].set_yscale('log', nonposy='clip')
    # ax[1][0].grid(color='m', which='major', axis='y', ls='-', lw=1)
    #
    # ax[1][1].boxplot(tn_feat)
    # ax[1][1].set_title('TN')
    # ax[1][1].set_yscale('log', nonposy='clip')
    # ax[1][1].grid(color='m', which='major', axis='y', ls='-', lw=1)


    #
    plt.tight_layout()
    #plt.show()
    plt.savefig(args.outfig, dpi=600)






