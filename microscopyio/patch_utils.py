import numpy as np
import pandas as pd
import os


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


def get_patch_metadata(in_file, o_file=None, in_limit=None):
    meta_tags = ['loc_x', 'loc_y', 'sz_x', 'sz_y', 'label', 'orig_file']
    instances = {'TU': [], 'BO': [], 'NO': []}

    skip_rows = 0

    with open(in_file) as csv_file:
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
                in_file,
                str(e)
            ))
            return None

    if o_file is not None and os.path.exists(o_file):
        if o_file[-4:] == ".csv":
            with open(o_file) as csv_ofile:
                odf = pd.read_csv(csv_ofile, delimiter=';', index_col=False, skipinitialspace=True)

                skip_rows = odf.shape[0]

    if in_limit is not None:
        len_tu = len(instances['TU'])
        len_bo = len(instances['BO'])
        len_no = len(instances['NO'])

        no_limit = in_limit
        if in_limit < 0:
            no_limit = len_no

        # all patches (including random selection of normal) computed, return empty
        if len_tu + len_bo + no_limit <= skip_rows:
            print('All patches already computed \t\t eta = 0:00:00 ')
            return None

        # some tumor patches not computed yet
        if len_tu > skip_rows:
            print('Reducing TU patches, some already computed')
            instances['TU'] = instances['TU'][skip_rows:]

        # all tumor computed, some border missing
        elif len_tu + len_bo > skip_rows:
            print('Removing TU and some BO, already computed')
            instances['TU'] = []
            instances['BO'] = instances['BO'][(skip_rows - len_tu):]

        # all tumor and all border computed
        else:
            print('Removing all TU+BO, already computed')
            instances['TU'] = []
            instances['BO'] = []
            instances['NO'] = instances['NO'][(skip_rows - len_tu - len_bo):]

    return instances
