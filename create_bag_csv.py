import sys

import numpy as np

from classification import csv_io


def get_selection_index(patches, max_patches, is_tumor=False, split_portions=(-1, 0.5, 0.5)):
    selection_index = []
    if is_tumor:
        tumor = [idx for idx, mp in enumerate(patches) if 'TU' in mp]
        border = [idx for idx, mp in enumerate(patches) if 'BO' in mp]
        other = [idx for idx, mp in enumerate(patches) if 'OT' in mp]

        # select all tumor patches for -1 or when less then t
        t_split = int(split_portions[0] * max_patches)
        rem = max_patches
        if split_portions[0] < 0 or len(tumor) < t_split:
            selection_index += tumor
            rem -= len(tumor)
        else:
            rem -= t_split
            selection_index += np.random.permutation(tumor)[:t_split].tolist()

        if (split_portions[1] + split_portions[2]) == 0:
            return selection_index

        rem = max(rem, 0)

        b_split = int(split_portions[1] * rem)
        o_split = max(rem - b_split, int(split_portions[2] * rem))
        if split_portions[1] < 0 or len(border) < b_split:
            selection_index += border
            rem -= len(border)
        else:
            rem -= b_split
            selection_index += np.random.permutation(border)[:b_split].tolist()

        if split_portions[2] < 0:
            o_split = len(other)
            rem = -1
        else:
            rem = o_split

        selection_index += np.random.permutation(other)[:max(rem, o_split)].tolist()

    else:
        selection_index = np.random.permutation(np.arange(len(patches)))[:max_patches]

    return selection_index


if __name__ == "__main__":

    input_file = sys.argv[1]
    max_patches = int(sys.argv[2])
    output_file = sys.argv[3]

    prop = [0, 0, 0]
    if len(sys.argv) > 4:
        inprop = list(map(float, sys.argv[4:]))

        for i, ip in enumerate(inprop):
            prop[i] += ip

    meta_names = ['loc_x', 'loc_y', 'label']

    is_tumor = False
    if 'tumor' in input_file:
        is_tumor = True

    study_features = ["sp", "rel_area_lumen", "rel_area_nuclei", "rel_area_stroma", "num_", "HIST"]
    all_patches, meta_patches, colnames = csv_io.import_csv_file_wnames(input_file=input_file,
                                                                        features=study_features,
                                                                        delim=';',
                                                                        skip_border=False)

    if all_patches is None or len(all_patches) < 1:
        quit()

    splits = tuple(prop)
    sel_idx = get_selection_index(meta_patches, max_patches, is_tumor, split_portions=splits)

    CSV = ";".join(map(str, meta_names)) + ";" + ";".join(map(str, colnames)) + "\n"
    for idx in sel_idx:
        CSV += ";".join(map(str, meta_patches[idx])) + ";" + ";".join(map(str, all_patches[idx])) + "\n"

    with open(output_file, 'w') as f:
        f.write(CSV)
    f.close()
