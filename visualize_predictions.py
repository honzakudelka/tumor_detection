import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import cv2
import sys
import pandas as pd

from glob import glob
from microscopyio.slide_image import NDPISlideImage, CamelyonSlideImage


def visualize_prediction( feature_name, pred_results, M, labels, train_len=0, output_prefix=""):

    data_counter = 0
    for m in M:
        metadata = m

        # mean_preds = np.sum(np.array(pred_results[alg_name])-np.array(labels), axis=0)
        scalars = np.abs(pred_results * 100. / np.max(pred_results))

        petacc_root = "/datagrid/Medical/microscopy/petacc3"
        image_name = os.path.basename(metadata[0])[:8]
        patch_size = 1024
        image_path = petacc_root + "/batch_2/" + image_name + ".ndpi"
        if not os.path.exists(image_path):
            image_path = petacc_root + "/batch_1/" + image_name + ".ndpi"

        si = NDPISlideImage(image_path, None)
        patchvis = si.get_patch_visualization(6, metadata[2], patch_size,
                                              scalars=scalars[m[3]:m[3]+len(m[2])],
                                              line_thickness=1,
                                              show=False, filled=True)

        # if data_counter < train_len:
        #     cv2.putText(patchvis, 'test', (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # else:
        #     cv2.putText(patchvis, 'vali', (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite("{}_{}_{}".format(output_prefix, image_name, feature_name) + ".png", patchvis)

        data_counter += 1


def visualize_prediction_came(feature_name, pred_results, M, labels, train_len=0, output_prefix=""):

    data_counter = 0
    validation_offset = 0
    for m in M:
        metadata = m

        scalars = np.abs(pred_results * 100. / np.max(pred_results))

        image_root = "/datagrid/Medical/microscopy/CAMELYON16/training"
        image_name = os.path.basename(metadata[0])
        patch_size = 1024  # int(os.path.basename(metadata[0]).split('_')[3][1:])

        image_name_parts = image_name.split('_')

        if 'tumor' in metadata[0]:
            image_root += "/tumor/t"
            image_path = image_root + "{}_{}.tif".format(image_name_parts[1][1:], image_name_parts[2])

        elif 'normal' in metadata[0]:
            image_root += "/normal/N"
            image_path = image_root + "{}_{}.tif".format(image_name_parts[1][1:], image_name_parts[2])
            continue

        else:
            image_root = "/datagrid/Medical/microscopy/CAMELYON16/testing/"
            image_path = image_root + "Test_{}.tif".format(image_name_parts[2])

        si = CamelyonSlideImage(image_path, None)
        print(" [VisOutput] Loading image {}".format(image_path))
        if not os.path.exists(image_path):
            print (" !! FAILED, IMAGE NOT FOUND !!")
            break

        patchvis = si.get_patch_visualization(6, metadata[2], patch_size,
                                              scalars=scalars[
                                                      validation_offset + m[3]:validation_offset + m[3] + len(
                                                          m[2])],
                                              line_thickness=1,
                                              show=False, filled=True)

        # if data_counter < train_len:
        #     cv2.putText(patchvis, 'test', (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # else:
        #     cv2.putText(patchvis, 'vali', (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite("{}_{}_{}".format(output_prefix, image_name, feature_name) + ".png", patchvis)

        data_counter += 1


def get_predictions(csvfile, algorithm, features):

    instances = []
    M = []
    labels = []

    label_dict = {-1: 'NO', 1: 'TU'}

    with open(csvfile) as csv_file:
        try:
            X = []
            y = []

            df = pd.read_csv(csv_file, delimiter=';', index_col=False, skipinitialspace=True)
            df = df.sort(columns='image')

            col_names = df.columns.values
            col_indices = []
            # filter input columns by prefix
            for feature_name in features:
                col_indices += [idx for idx, name in enumerate(col_names)
                                if name.startswith(feature_name)]

            meta_names = ['locx', 'locy', 'label']
            meta_indices = []
            for meta_name in meta_names:
                meta_indices.append(np.where(col_names == meta_name)[0][0])

            imname_idx = np.where(col_names == 'image')[0][0]
            imname = ''

            last_start = 0
            for row in df.itertuples():

                if not imname == row[imname_idx+1]:

                    if len(X) > 0:
                        instances += X
                        labels += y
                        M.append((imname, -1, meta_instances, last_start ))

                    # reset
                    last_start += len(X)
                    imname = row[imname_idx + 1]
                    meta_instances = []
                    X = []
                    y = []

                row_arr = np.asarray([row[i + 1] for i in col_indices])
                meta_arr = np.asarray([row[i + 1] for i in meta_indices])

                y.append(meta_arr[2])
                X.append(row_arr)
                meta_instances.append(((int(meta_arr[0]),
                                        int(meta_arr[1])), label_dict[meta_arr[2]]))

        except ValueError as e:
            print("Failed to parse file {0} \n Exception: \n ----------- \n {1}".format(
                csv_file,
                str(e)
            ))
            return None

    return instances, labels, M


if __name__ == "__main__":

    study_root = "/datagrid/personal/herinjan/milearning/microscopy-python/features/training"
    study_groups = ["petacc"]
    test_root = "/datagrid/personal/herinjan/milearning/microscopy-python/features/validation"
    study_labels = [-1, 1]
    group_dict = dict(zip(study_groups, study_labels))
    parse_str = "E*s1024_pext_0_stru_his_txt_features.csv"
    prediction_file = None

    if len(sys.argv) > 1:
        prediction_file = sys.argv[1]    # "*s1024_pext_0_stru_his_txt_features.csv"
        output_prefix = sys.argv[2]

    visualize_came = False
    if len(sys.argv) > 3 and sys.argv[3] == 'camelyon':
        visualize_came = True

    #algorithms = ['MI-SVM', 'MI-SVM_k_opt', 'top-MI-T-opt+cont_loss+bag', 'top-MI-SVM_k=5_all']
    #algorithms = ['MIL-ARF', 'MIL-RF', 'T-MIL-ARF', 'T-bag-RF+opt+bag', 'aMIL-RF+k_opt', 'aMIL-RF', 'aT-MIL-ARF']
    algorithms = ['aMIL-ARF+k_opt']

    #algorithms = ['instance MIL-RF']

    if not os.path.exists(os.path.dirname(output_prefix)):
        os.mkdir(os.path.dirname(output_prefix))

    for algorithm in algorithms:

        features = ["{}_mean".format(algorithm)]

        X, y, M = get_predictions(prediction_file, algorithm, features)

        for idx in range(len(features)):
            pred_results = np.array(X)[:, idx]

            if visualize_came:
                visualize_prediction_came(
                    feature_name=features[idx],
                    pred_results=pred_results, M=M, labels=y,
                    train_len=len(M),
                    output_prefix=output_prefix
                )

            else:

                visualize_prediction(feature_name=features[idx],
                                     pred_results=pred_results, M=M, labels=y,
                                     train_len=len(M),
                                     output_prefix=output_prefix
                                     )

