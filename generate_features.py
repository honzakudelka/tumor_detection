import argparse
import os
import random
import sys
from datetime import timedelta
from timeit import default_timer as timer

import cv2
import numpy as np
import pandas as pd

from feature_extraction import structural, textural
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generate synthetic feature data in Camelyon style")
    parser.add_argument('-i','--input', type=str, help='Input file (fullpath)')
    parser.add_argument('-o', '--output', type=str, help='Output file (fullpath)')
    parser.add_argument('-mp', '--max_patches', type=int, help='Limit number of patches (select random if less then available, default=-1',
                        default=-1)
    parser.add_argument('-l', '--extract_level', type=int, help='Patch extraction level',
                        default=0)

    in_args = parser.parse_args()

    input_file = in_args.input
    output_file = in_args.output
    max_no_patches = in_args.max_patches
    patch_extract_level = in_args.extract_level

    patches_in = get_patch_metadata(input_file, output_file)

    if patches_in is None or (len(patches_in['NO']) + len(patches_in['TU']) + len(patches_in['BO'])) < 1:
        print("Nothing to do for input directory " + input_file)
        quit()

    print("Retrieved: {} Normal, {} Tumor and {} Border patches".format(len(patches_in['NO']),
                                                                        len(patches_in['TU']),
                                                                        len(patches_in['BO']))
          )

    patches = []
    patches = patches_in['TU'] + patches_in['BO'] + patches_in['NO']
    # patches += patches_in['TU'] + patches_in['BO']

    visited_no_patches = 0
    if len(patches) < 1:
        print("Nothing to do for input directory " + input_file)
        quit()

    image_features = []
    n = len(patches)
    i = 1
    avg_time = 0
    cum_time = 0

    mv_len = 4
    normal_mv_means = 2.0 * np.array([0.0, 1.0, -1.0, 0.2])
    tumor_mv_means = 2.0 * np.array([0.2, 0.8, -0.2, 1.0])
    normal_cov_scale = 0.1
    tumor_cov_scale = 0.1
    unit_cov = np.identity(mv_len)

    for patch_descriptor in patches:
        location = patch_descriptor[0]
        p_size = patch_descriptor[1]
        label = patch_descriptor[2]

        print("[{}] {} @ {}".format(label, p_size, location))
        t_start = timer()

        # Patch features generation
        generated_features = {}

        x = np.zeros(mv_len)
        x_no = np.random.multivariate_normal(normal_mv_means, cov=normal_cov_scale * unit_cov)
        x_tu = np.random.multivariate_normal(tumor_mv_means, cov=tumor_cov_scale * unit_cov)

        if label == 'NO':
            x = x_no
        elif label == 'TU':
            x = x_tu
        elif label == 'BO':
            x = 0.5 * (x_no + x_tu)
        else:
            gamma = 0.01 * np.random.randint(0, 100)
            x = gamma * x_no + (1-gamma) * x_tu

        for x_i, x_val in enumerate(x):
            generated_features.update({'x_val_{:02d}'.format(x_i): x_val})

        image_features.append(dict({"loc_x": location[0], "loc_y": location[1], "label": label},
                                   **generated_features))

        t_end = timer()

        cum_time += t_end - t_start
        avg_time = cum_time / (1. * i)

        eta = (n - i) * avg_time

        eta_str = timedelta(seconds=eta)
        print(" ...processed patch {0:03d}/{1} in {2:.3f} [s] \t eta = {3}".format(i, n, t_end - t_start, eta_str))
        i += 1

        # Terminate job after elapsed cumulative time, it must be < 4h when spawn into the SGE's fastjobs queue
        if cum_time > 120 * 60:
            break

    print_dictarr_to_csv(image_features, output_file, append=os.path.exists(output_file))
