from glob import glob

import numpy as np
import pandas as pd


def import_csv_file_wnames(input_file, features, delim=';', skip_border=False, skip_list=('BO', 'OT')):

    instances = []
    meta_instances = []
    ocol_names = []

    with open(input_file) as csv_file:
        try:
            print("[Input] {}".format(csv_file))
            df = pd.read_csv(csv_file, delimiter=delim, index_col=False,
                             skipinitialspace=True)

            col_names = df.columns.values
            col_indices = []
            # filter input columns by prefix
            for feature_name in features:
                col_indices += [idx for idx, name in enumerate(col_names)
                                if name.startswith(feature_name)]

            col_indices.sort()
            for idx in col_indices:
                ocol_names.append(col_names[idx])

            meta_names = ['loc_x', 'loc_y', 'label']
            meta_indices = []
            for meta_name in meta_names:
                meta_indices.append(np.where(col_names == meta_name)[0][0])

            for row in df.itertuples():
                row_arr = np.asarray([row[i + 1] for i in col_indices])
                meta_arr = np.asarray([row[i + 1] for i in meta_indices])
                try:
                    if not np.isfinite(row_arr.sum()):
                        print(" :: Skipping line with NaN values")
                        continue
                except TypeError as e:
                    print("Caught exception while importing row {} \n Exception \t {}".format(i, e))
                    continue

                    # if skip_border and (('BO' in meta_arr) or ('OT' in meta_arr)):
                if skip_border and (meta_arr[2] in skip_list):
                    # print(" :: Skipping undecided tumor patch")
                    continue

                instances.append(row_arr)
                meta_instances.append(meta_arr)

        except ValueError as e:
            print("Failed to parse file {0} \n Exception: \n ----------- \n {1}".format(
                input_file,
                str(e)
            ))
            return None

    return instances, meta_instances, ocol_names


def import_csv_file(input_file, features, delim=';', skip_border=False, skip_list=('BO', 'OT'),
                    label_dict=None, printed=True, output_skip=False):

    instances = []
    meta_instances = []
    labels = []
    valid_lines = []

    if label_dict is None:
        label_dict = {'NO': -1, 'TU': 1, 'BO': 1, 'OT': 1}

    with open(input_file) as csv_file:
        try:

            df = pd.read_csv(csv_file, delimiter=delim, index_col=False,
                             skipinitialspace=True)

            col_names = df.columns.values
            col_indices = []
            # filter input columns by prefix
            for feature_name in features:
                col_indices += [idx for idx, name in enumerate(col_names)
                                if name.startswith(feature_name)]

            col_indices.sort()
            if not printed:
                print("[FEATURE NAMES]")
                name_str = ""
                for idx in col_indices:
                    name_str += "{};".format(col_names[idx])

                print(name_str)

            meta_names = ['loc_x', 'loc_y', 'label']
            meta_indices = []
            for meta_name in meta_names:
                meta_indices.append(np.where(col_names == meta_name)[0][0])

            for row in df.itertuples():
                row_arr = np.asarray([row[i + 1] for i in col_indices])
                meta_arr = np.asarray([row[i + 1] for i in meta_indices])

                if not len(row_arr) == len(col_indices):
                    continue

                # if skip_border and (('BO' in meta_arr) or ('OT' in meta_arr)):
                if skip_border and (meta_arr[2] in skip_list):
                    # print(" :: Skipping undecided tumor patch")
                    continue

                try:
                    if not np.isfinite(row_arr.sum()):
                        if not output_skip:
                            continue
                        else:
                            valid_lines.append(-1)
                        
                    else:
                        if output_skip:
                            valid_lines.append(1)
                except TypeError as e:
                    print("Caught exception while importing row \n Exception \t {}".format(e))
                    print(row_arr)
                    print(";".join(map(str, row_arr)))
                    if not output_skip:
                        continue
                    else:
                        valid_lines.append(-1)


                labels.append(label_dict[meta_arr[2]])
                instances.append(row_arr)
                meta_instances.append(((int(meta_arr[0]),
                                        int(meta_arr[1])), meta_arr[2]))

        except ValueError as e:
            print("Failed to parse file {0} \n Exception: \n ----------- \n {1}".format(
                input_file,
                str(e)
            ))
            return None

    if not output_skip:
        return instances, meta_instances, labels
    else:
        return instances, meta_instances, labels, valid_lines


def import_study_multiple_per_subject(features, delim=';',
                                      skip_list=('BO', 'OT'), label_dict=None, file_list=None, n_merge=2):
    X = []
    M = []
    y = []

    if file_list is None:
        raise RuntimeError("Must provide a file list as tuple per subject.")

    skip_undecided = False
    if len(skip_list):
        skip_undecided = True

    group_label = -1
    last_start = 0
    dtol = 4

    for input_file_tuple in file_list:
        if not len(input_file_tuple) == n_merge:
            print("[WARNING] Different count than {} provided".format(n_merge)
                                 + input_file_tuple)

            continue

        instances, meta_instances, labels, skips = [], [], [], []
        current_len = -1
        all_retrieved = True
        for input_file in input_file_tuple:
            print("Loading from file {}".format(input_file))
            f_instances, f_meta_instances, f_labels, f_skip = import_csv_file(input_file, features, delim,
                                                                              skip_undecided, skip_list, label_dict, printed=True,
                                                                              output_skip=True)

            if abs(len(f_labels) - current_len) > dtol-1:
                if current_len < 0:
                    current_len = len(f_labels)
                else:
                    print("[WARNING] Retrieved {} instances from file {}, exepected {}".format(
                        len(f_labels),
                        input_file,
                        current_len
                    ))
                    all_retrieved = False
                    break

            diff = abs(len(f_labels) - current_len)
            if 0 < diff < dtol:

                f_instances = f_instances[:-diff]
                f_meta_instances = f_meta_instances[:-diff]
                f_labels = f_labels[:-diff]
                f_skip = f_skip[:-diff]

            instances.append(f_instances)
            meta_instances.append(f_meta_instances)
            labels.append(f_labels)
            skips.append(f_skip)

        if not all_retrieved:
            continue

        merged_skips = np.array(skips)
        valid_lines = np.where( np.sum(merged_skips, axis=0) == len(input_file_tuple) )

        # merge results for tuple
        # merged_instances = np.array(instances)
        merged_instances = np.concatenate(instances, axis=1)

        X += [minst for minst in merged_instances]
        y += [labels[0][i] for i in valid_lines[0]]
        mi = [meta_instances[0][i] for i in valid_lines[0]]
        M.append((input_file_tuple[0], group_label, mi, last_start))
        last_start += len(merged_instances)
        print("     + {} instances".format(current_len))

    return X, y, M


def import_study_from_csv(study_root, groups, group_dict, features, parse_string, delim=';',
                          skip_list=('BO', 'OT'), label_dict=None, file_list=None, restrict_size=False,
                          max_patches=100):

    X = []
    M = []
    y = []

    last_start = 0

    if file_list is not None:
        group_label = -1

        skip_undecided = False
        if len(skip_list):
            skip_undecided = True

        printed = False

        for input_file in file_list:
            print("Loading from file {}".format(input_file))
            instances, meta_instances, labels = import_csv_file(input_file, features, delim,
                                                                skip_undecided, skip_list, label_dict, printed=printed)

            if instances is None:
                continue

            if not printed:
                printed = True

            if restrict_size:
                n_labels = np.where(np.array(labels) < 0)
                p_labels = np.where(np.array(labels) > 0)

                if 'normal' in input_file:
                    extract_len = min(len(n_labels[0]), max_patches)

                if len(p_labels[0]) < 1 or len(p_labels[0]) < len(n_labels[0]):
                    np.random.shuffle(n_labels[0])
                    n_sel = n_labels[0]
                    index_list = np.concatenate((p_labels[0], n_sel[:extract_len]))

                    instances = [instances[i] for i in index_list]
                    meta_instances = [meta_instances[i] for i in index_list]
                    labels = [labels[i] for i in index_list]

            X += instances
            y += labels
            M.append((input_file, group_label, meta_instances, last_start))
            last_start += len(instances)
            print("     + {} instances".format(len(instances)))
    else:
        for group in groups:
            group_root = study_root+"/"+group
            group_label = group_dict[group]

            input_csv_files = sorted(glob(group_root+'/'+parse_string))
            skip_undecided = False
            if len(skip_list):
                skip_undecided = True

            printed = False

            for input_file in input_csv_files:

                instances, meta_instances, labels = import_csv_file(input_file, features, delim,
                                                                    skip_undecided, skip_list, label_dict, printed=printed)

                if instances is None:
                    continue

                if not printed:
                    printed = True

                if equal_sizes:
                    n_labels = np.where(labels < 0)
                    p_labels = np.where(labels > 0)

                    if len(n_labels) < len(p_labels):
                        np.random.shuffle(p_labels)
                        index_list = np.concatenate((n_labels, p_labels[:len(n_labels)]))

                        instances = [instances[i] for i in index_list]
                        meta_instances = [meta_instances[i] for i in index_list]
                        labels = [labels[i] for i in index_list]

                X += instances
                y += labels
                M.append((input_file, group_label, meta_instances, last_start))
                last_start += len(instances)

    return X, y, M
