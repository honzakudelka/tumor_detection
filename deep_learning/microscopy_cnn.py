"""
File:   deep_learning.microscopy_cnn.py

Application of a CNN architecture to microscopic data. The script loads the description of available patches
from CSV files. With these, the microscopy_dataset provides a dataloader interface to tflearn
and can thus be taken for on-demand patch loading from WSI images.

Important parameters except I/O :

    --architecture      the selected CNN architecture
    --cnn_options       list of options to control some additional functionality

                        BNorm       -  use batch normalization

                        HEPlain     -  classify with H&E channels
                        ODensPlain  -  classify in the optical density space
                        ColorTra    -  add layers at the beginning for learning a color transform
                        ODensColor  -  transform to optical density and add color transformation layers


    --patch_size        size of the patches to be extracted as input patches of the network
    --original_size     size of patches stored in the description files, if patch_size divides original_size
                        multiple (non-overlapping) patches are extracted from a single input patch

    --callbacks         additional callbacks to be executed each given step, currently only AUCPredict available,


Author:
    Jan Hering (BIA/CMP)
    jan.hering@fel.cvut.cz

"""
import argparse
import os
from glob import glob

import numpy as np
import tflearn
from tflearn.layers.core import input_data
from tflearn.layers.estimator import regression
from tflearn.optimizers import Momentum
from tflearn.utils import feed_dict_builder

from deep_learning.cnn_architectures import create_vgg16_network, create_simple_network
from deep_learning import microscopy_dataset
from deep_learning import micro_augmentation
from deep_learning.cnn_callbacks import PredictionCallback
from classification import csv_io


def _is_valid_size(network_in, patch_size, patch_level):

    real_extracted_size = patch_size // 2**patch_level
    return network_in < real_extracted_size


def _join_and_relabel_groups( X, y, M, group_labels=None):
    """Join instances (X), labels (y) and meta-data (M) from multiple groups.
    
    :param group_labels if specified, the values in y are relabeled by the given labels
    """
    Xn = []
    yn = []
    Mn = []
    
    offset = 0
    for i in range(len(X)):
        Xin = X[i]
        yin = y[i]
        Min = M[i]
        
        for mi in range(len(Min)):
            Mt_entry = Min[mi]
            
            Xn += Xin[Mt_entry[3]:Mt_entry[3] + len(Mt_entry[2])]
            
            ygr = yin[Mt_entry[3]:Mt_entry[3] + len(Mt_entry[2])]
            if group_labels is not None:
                ygr = [group_labels[i] for lx in range(len(ygr))]
               
            Mn.append((Mt_entry[0], Mt_entry[1], Mt_entry[2][:], offset))
            yn += ygr
            offset += len(Mt_entry[2])
    
    return Xn, yn, Mn


def _train_test_split(M, tr_idx, te_idx, y_in=None, y_train=None, y_test=None):
    """
    Split the vector with meta-data into two lists according to train / test idx, recomputes the offsets

    :param M: Complete list of meta-data
    :param tr_idx: Indices included in the training set
    :param te_idx: Indices included in the testing set
    :return: Mt, Mtest: Two lists with separated metadata
    """

    Mtrain = []
    Mtest = []

    offset = 0
    for idx in tr_idx:
        Mt_entry = M[idx]

        Mtrain.append((Mt_entry[0], Mt_entry[1], Mt_entry[2][:], offset))
        offset += len(Mt_entry[2])

        if y_in is not None:
            y_train += y_in[Mt_entry[3]:Mt_entry[3] + len(Mt_entry[2])]

    offset = 0
    for idx in te_idx:
        Mt_entry = M[idx]

        Mtest.append((Mt_entry[0], Mt_entry[1], Mt_entry[2][:], offset))
        offset += len(Mt_entry[2])

        if y_in is not None:
            y_test += y_in[Mt_entry[3]:Mt_entry[3] + len(Mt_entry[2])]

    return Mtrain, Mtest
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Deep-learning for microscopic images")
    input_group = parser.add_argument_group('I/O Group')
    input_group.add_argument('-sr', '--study_root', type=str, help='Directory root for training data',
                             default='/datagrid/personal/herinjan/milearning/microscopy-python/features/training')
    input_group.add_argument('-sg', '--study_groups', type=str, nargs='+', help='Groups (subdirs in the study root)',
                             required=True)
    input_group.add_argument('-gl', '--group_labels', type=int, nargs='+', help='Group labels (same order as '
                                                                                'study_groups)',
                             required=True)

    input_group.add_argument('-ps', '--parse_string', type=str, help='Regex to retrieve relevant data',
                             default='*s1024_pext_0_stru_his_txt_features.csv')

    network_group = parser.add_argument_group('CNN Architecture Settings')
    network_group.add_argument('-id', '--run_id', type=str,
                               help='Unique run ID for output (saving / tensorboard report)',
                               required=True)
    network_group.add_argument('-opt', '--cnn_options', type=str, nargs='*', help='Specify additional processing options')
    network_group.add_argument('-a', '--architecture', type=str, help='Network architecture',
                               required=True)

    data_group = parser.add_argument_group('CNN Data Handling')
    data_group.add_argument('-s', '--patch_size', type=int, default=1024,
                            help='Size of patches to be extracted from the WSI image, if multiple of the original '
                                 'size several patches will be subsampled from a single input patch region')
    data_group.add_argument('-os', '--original_size', type=int, default=None,
                            help='Size of patches in the meta-data')
    data_group.add_argument('-l', '--extract_level', type=int,
                            help='Resolution level for extracted patches',
                            default=2)
    data_group.add_argument('-nc', '--n_classes', type=int, help='Number of tissue classes \n 2 - binary '
                                                                 'Normal/Tumor, 3 - Normal/TumorA/TumorB',
                            default=2)

    other_group = parser.add_argument_group('Other settings')
    other_group.add_argument('-nrep', '--number_repetitions', type=int, default=5,
                             help='Number of cross-validation runs')
    other_group.add_argument('-tts', '--train_split', type=float, default=0.2,
                             help='Relative size of the test split, train size is 1-train_split')
    other_group.add_argument('-cb', '--callbacks', type=str, nargs='*', help='Activate callbacks (AUCPrediction)')

    in_args = parser.parse_args()

    # Data I/O - read metadata extracted from the images
    label_dict = {'NO': 0, 'TU': 1, 'BO': 1, 'OT': 1}
    study_features = ['loc_x', 'loc_y']

    group_dict = dict(zip(in_args.study_groups, in_args.group_labels))
    
    allX = []
    ally = []
    allM = []
    n_classes = in_args.n_classes
    for group in in_args.study_groups:

        # Here, we skip also normal patches, we want to retrieve only tumor patches
        input_csv_files = sorted(glob(os.path.join(in_args.study_root, group, in_args.parse_string)))
        Xgr, ygr, Mgr = csv_io.import_study_from_csv(study_root=in_args.study_root,
                                                     group_dict=group_dict, groups=in_args.study_groups,
                                                     features=study_features, parse_string=in_args.parse_string,
                                                     label_dict=label_dict, skip_list=('BO', 'OT'),
                                                     file_list=input_csv_files)
        
        allX.append(Xgr)
        ally.append(ygr)
        allM.append(Mgr)

        # read tumor-class from the second group with different label
        if n_classes > 2:
            label_dict = {'NO': 0, 'TU': 2, 'BO': 2, 'OT': 2}

    X, y, M = _join_and_relabel_groups(allX, ally, allM, group_labels=None)

    # Global settings for the network
    ps = in_args.patch_size
    if in_args.original_size is None:
        oris = ps
    else:
        oris = in_args.original_size

    # FIXME Pre-computed mean/stdev for PETACC patches
    c_mean = np.array([200, 150, 200], dtype=np.float32)
    c_stdev = np.array([32.0, 25.5, 34.0], dtype=np.float32)
    
    # Setup preprocessing
    extract_level = 0
    apply_colortransform = False
    transform_to_odensity = False
    apply_batchnorm = False
    apply_bg_subtraction = False
    apply_he_augmentation = False
    convert_to_he = False
    convert_to_he_rand = False

    if 'BNorm' in in_args.cnn_options:
        apply_batchnorm = True
    if 'ColorTra' in in_args.cnn_options:
        apply_colortransform = True
    if 'ODensColor' in in_args.cnn_options:
        transform_to_odensity = True
        apply_colortransform = True
    if 'ODensPlain' in in_args.cnn_options:
        transform_to_odensity = True
    if 'BGSub' in in_args.cnn_options:
        apply_bg_subtraction = True
    if 'HEAug' in in_args.cnn_options:
        apply_he_augmentation = True
    if 'HEColor' in in_args.cnn_options:
        convert_to_he = True
        apply_colortransform = True
    if 'HEPlain' in in_args.cnn_options:
        convert_to_he = True
    if 'HERand' in in_args.cnn_options:
        convert_to_he_rand = True

    # ---- CNN Configuration ----
    # (1) preprocessing
    img_prep = micro_augmentation.HEPreprocessing()
    if apply_bg_subtraction:
        img_prep.subtract_background()

    if transform_to_odensity:
        img_prep.add_odens_transform(scale=255)

    if convert_to_he:
        img_prep.convert_to_he_space()

    if convert_to_he_rand:
        img_prep.convert_to_he_space_randsep()

    if not convert_to_he:
        # img_prep.add_featurewise_zero_center(mean=c_mean)
        # img_prep.add_featurewise_stdnorm(std=c_stdev)
        img_prep.add_samplewise_zero_center()
        img_prep.add_samplewise_stdnorm()

    # (2) Augmentation
    img_aug = micro_augmentation.HEAugmentation()

    if apply_he_augmentation:
        img_aug.add_random_he_variation(alpha=0.015, mean=c_mean, stdev=c_stdev)

    img_aug.add_random_flip_leftright()
    img_aug.add_random_90degrees_rotation(rotations=[0, 1])

    # (3) Input formatting and network creation
    model = None
    if in_args.architecture == 'VGG16':

        if not _is_valid_size(224, in_args.patch_size, in_args.extract_level):
            raise RuntimeError("Real size of extracted patch less than network input size!")

        img_prep.add_random_crop((224, 224))
        cnn_network = input_data(shape=[None, 224, 224, 3],
                                 data_preprocessing=img_prep,
                                 data_augmentation=img_aug)

        cnn_network = create_vgg16_network( cnn_network, num_classes=n_classes,
                                            normalize_batch=apply_batchnorm,
                                            add_color_transfer=apply_colortransform)

        MomOpt = Momentum(learning_rate=0.01, lr_decay=0.8, decay_step=200)
        cnn_network = regression(cnn_network, optimizer=MomOpt,
                                 loss='categorical_crossentropy')

    elif in_args.architecture == 'simple':

        if not _is_valid_size(28, in_args.patch_size, in_args.extract_level):
            raise RuntimeError("Real size of extracted patch less than network input size!")

        img_prep.add_random_crop((28, 28))
        cnn_network = input_data(shape=[None, 28, 28, 3],
                                 data_preprocessing=img_prep,
                                 data_augmentation=img_aug)

        cnn_network = create_simple_network( cnn_network, num_classes=n_classes)

        cnn_network = network = regression(cnn_network, optimizer='adam',
                                           loss='categorical_crossentropy',
                                           learning_rate=0.001)

    else:
        raise RuntimeError("Given architecture {} not supported yet.".format(in_args.architecture))

    # Create random splits for repeated testing
    t_indices = np.arange(0, len(M))
    t_split = int(in_args.train_split * len(t_indices))
    for rep_i in range(in_args.number_repetitions):

        # get random splits
        np.random.shuffle(t_indices)
        train_idx = t_indices[:-t_split]
        test_idx = t_indices[-t_split:]

        y_train = []
        y_test = []

        M_train, M_test = _train_test_split(M, train_idx, test_idx, y, y_train, y_test)

        X, Y = microscopy_dataset.microscopy_preloader(meta_data_list=M_train,
                                                       extract_level=in_args.extract_level,
                                                       original_size=(oris, oris),
                                                       extract_size=(ps, ps),
                                                       n_classes=n_classes,
                                                       label_dictionary=None,
                                                       labels=y_train)
        valX, valY = microscopy_dataset.microscopy_preloader(meta_data_list=M_test,
                                                             extract_level=in_args.extract_level,
                                                             original_size=(oris, oris),
                                                             extract_size=(ps, ps),
                                                             n_classes=n_classes,
                                                             label_dictionary=None,
                                                             labels=y_test)

        # Training
        str_run_id = in_args.run_id + "_rep_{:02d}".format(rep_i)
        model = tflearn.DNN(cnn_network,
                            max_checkpoints=1, tensorboard_verbose=1,
                            tensorboard_dir="/datagrid/personal/herinjan/halmos_tmp/tflearn_logs",
                            checkpoint_path="/datagrid/personal/herinjan/halmos_tmp/tflearn_snapshot" + str_run_id )

        # Handle callbacks
        model_callbacks = []
        if 'AUCPredict' in in_args.callbacks:
            # Own prediction callback
            tf_predictor = model.predictor
            validationCb = PredictionCallback(predictor=tf_predictor,
                                              t_inputs=model.inputs,
                                              data=valX,
                                              labels=y_test,
                                              validation_step=250,
                                              session=model.session,
                                              batch_size=128)

            model_callbacks.append(validationCb)

        model.fit(X, Y, n_epoch=3, shuffle=True, validation_set=(valX, valY),
                  show_metric=True, batch_size=32, snapshot_step=1000,
                  snapshot_epoch=False, run_id=str_run_id, callbacks=model_callbacks
                  )

        model.predict(valX[:])
        
        
        
        
    
    








