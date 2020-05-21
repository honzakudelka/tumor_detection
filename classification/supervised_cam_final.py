import argparse
import csv
import os
from glob import glob
from scipy import interp

import cv2
import matplotlib as mpl
import numpy as np
# classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier

from classification import csv_io
from microscopyio.slide_image import CamelyonSlideImage

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt


base_fpr = np.linspace(0, 1, 101)

validation_runs_meanroc = []
testing_runs_meanroc = []


def _get_simple_csv():
    tempX = []
    X = []
    y = []

    file = open("/local/herinjan/Development/microscopy-io/ADCC/CLASSIF/data.csv", "r")
    train = csv.reader(file)
    for row in train:
        tempX.append(row[1:20])
        if row[0] == 'T':
            y.append(1)
        else:
            y.append(-1)
    for row in tempX:
        X.append([float(i) for i in row])

    return X, y


def _hyperparam_rand_opt(clf, param_dist, X, y, train_idx, test_idx, n_iter=20):

    rs = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, scoring='average_precision',
                            error_score=0., n_iter=n_iter, n_jobs=3, verbose=10)
    rs.fit(X=X[train_idx], y=y[train_idx])

    print("RandomizedSearch Result")
    print(rs.best_params_)

    final_clf = rs.best_estimator_

    final_clf.fit(X[train_idx], y[train_idx])
    preds = final_clf.predict(X[test_idx])
    p_preds = None

    fpredict = getattr(final_clf, "predict_proba", None)
    if fpredict is None:
        fpredict = getattr(final_clf, "decision_function", None)

    if callable(fpredict):
        p_preds = fpredict(X[test_idx])

    # validation_accuracy = np.average(preds == y[validation_idx])
    #
    # print("\t Accuracy: {0: 0.4f}".format(validation_accuracy))

    return preds, p_preds, final_clf


def _hyperparam_opt(clf, params, X, y, train_idx, validation_idx):

    need_search = False
    for x in params:
        if len(params[x]) > 1:
            need_search = True
            break

    if need_search:
        gs = GridSearchCV(estimator=clf,
                          param_grid=params, scoring='f1_weighted', error_score=0)
        gs.fit(X=X[train_idx], y=y[train_idx])

        print("GridSearch Result")
        print(gs.best_params_)

        final_clf = gs.best_estimator_
    else:
        final_clf = clf

        # pass the parameters to the classifier
        for parameter, value in params.items():

            if hasattr(final_clf, parameter):
                # de-capsulate the input parameters (given as list in the params dict)
                setattr(final_clf, parameter, value[0])

    print("Fitting with parameters: \n" + str(final_clf.get_params()))

    with open('/tmp/cam16_bestparams.log', 'a') as fh:
        fh.write( str(final_clf.get_params()) + "\n")

    fh.close()

    final_clf.fit(X[train_idx], y[train_idx])
    preds = final_clf.predict(X[validation_idx])
    p_preds = None

    fpredict = getattr(final_clf, "predict_proba", None)
    if fpredict is None:
        fpredict = getattr(final_clf, "decision_function", None)

    if callable(fpredict):
        p_preds = fpredict(X[validation_idx])

    # validation_accuracy = np.average(preds == y[validation_idx])
    #
    # print("\t Accuracy: {0: 0.4f}".format(validation_accuracy))

    return preds, p_preds, final_clf


def auc_analysis(f, lab, pos_label=1, suffix="", ax=None, tprs=None):
    ft = f[:]
    if len(f.shape) > 1:
        ft = f[:, 1]

    fpr, tpr, roc_thr = roc_curve(lab, ft, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    n_test_labels = np.array(lab)
    selection = (1 - tpr) * np.sum(n_test_labels[np.where(n_test_labels > 0)]) + \
                fpr * (-1 * np.sum(n_test_labels[np.where(n_test_labels < 0)]))

    roc_idx = np.argmin(fpr - tpr)

    a_sens = 1 - fpr
    a_spec = tpr

    if ax is not None:
        ax.plot(fpr, tpr, label='ROC-AUC = {0:.3f}, {1}'.format(roc_auc, suffix), alpha=0.25)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])

        if tprs is not None:
            tpr_i = interp(base_fpr, fpr, tpr)
            tpr_i[0] = 0.0
            tprs.append(tpr_i)

    roc_idx2 = np.argmax(a_sens + a_spec)

    spec = 1.0 - fpr[roc_idx]
    sens = tpr[roc_idx]
    thr_ = roc_thr[roc_idx]

    return roc_auc, spec, sens


def analyze_bag_results( algorithms, pred_results, M ):

    for algorithm in algorithms:

        a_name = algorithm[2]
        a_preds = pred_results[a_name]
        b_labs = [m[1] for m in M if len(m[2]) > 0]
        bag_f1 = []
        bag_auc = []

        for a in a_preds:
            b_preds = []
            b_posi = []
            for i, m in enumerate(M):
                if len(m[2]) == 0:
                    continue

                if i == len(M)-1:
                    a_bag = a[M[i][3]:]
                else:
                    a_bag = a[M[i][3]:M[i+1][3]]

                b_preds.append(np.max(a_bag))
                b_posi.append(np.sum(a_bag[a_bag > 0])/(1.*len(a_bag)))

            bag_f1.append(f1_score(y_pred=b_preds, y_true=b_labs, pos_label=1))
            bag_auc.append( np.array( auc_analysis(np.array(b_posi), b_labs)))

        bag_f1 = np.array(bag_f1, dtype='float32')

        perf_str = ''
        for i in range(3):
            ares = [item[i] for item in bag_auc]
            mean_perf = np.mean(ares, axis=0)
            sd_perf = np.std(ares, axis=0)

            perf_str += "{0:0.4f}+-{1:0.4f} \t".format(mean_perf, sd_perf)

        print("Bag f1: {} : {} +- {} \n AUC/SPEC/SENS {}".format(a_name, np.mean(bag_f1), np.std(bag_f1), perf_str))


def cv_eval_classifier(X, y, Xtest, ytest, algorithms, rounds=5, feature_str="", output_prefix="", output_proba=False):
    global validation_runs_meanroc
    global testing_runs_meanroc
    
    y = [y for i, y in enumerate(y) if len(X[i]) == len(X[0])]
    X = [x for x in X if len(x) == len(X[0])]

    X = np.array(X)
    y = np.array(y)

    Xtest = np.array(Xtest)

    assert len(X) == len(y), "Instance and label array do not match... {0} != {1}".format(
        len(X), len(y)
    )

    print("CV Eval classifier: \n ---------- \n Data statistics: \n" +
          "Label: [-1] \t {}\n".format(len(y[y < 0])) +
          "Label: [ 1] \t {}".format(len(y[y > 0])))

    M = MinMaxScaler()
    X = M.fit_transform(X)
    Xtest = M.transform(Xtest)
    stratified_sampler = StratifiedShuffleSplit(n_splits=rounds, test_size=0.2)

    predict_result = {}

    result = {}

    mean_feature_performances = np.zeros(len(X[0]))
    for algorithm in algorithms:
        result.update({algorithm[2]: []})
        predict_result.update({algorithm[2]:  []})

    run_id = 0
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 10), dpi=150)
    fig.suptitle("Receiver-operating characteristics [Train set] \n Used features: "+feature_str)
    ax1.set_title('[Train set]')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')

    ax2.set_title('[Test set]')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')

    tprs = [[], []]

    for train_index, test_index in stratified_sampler.split(X, y):

        for algorithm in algorithms:

            # print(" :: {} ::".format(algorithm[2]))
            if len(algorithm) > 3 and algorithm[3] == 'RAND':
                preds, ppreds, final_clf = _hyperparam_rand_opt(clf=algorithm[0], param_dist=algorithm[1],
                                                                X=X, y=y, train_idx=train_index, test_idx=test_index, n_iter=30)
            else:
                preds, ppreds, final_clf = _hyperparam_opt(clf=algorithm[0], params=algorithm[1],
                                                       X=X, y=y, train_idx=train_index, validation_idx=test_index)

            train_values = auc_analysis(ppreds, y[test_index], suffix=" CV Run {}".format(run_id), ax=ax1, tprs=tprs[0])
            mean_feature_performances = mean_feature_performances + (1.0/rounds) * final_clf.feature_importances_

            # final clf predict
            tp_preds = None

            fpredict = getattr(final_clf, "predict_proba", None)
            if fpredict is None:
                fpredict = getattr(final_clf, "decision_function", None)

            if callable(fpredict):
                tp_preds = fpredict(Xtest)

            test_values = auc_analysis(tp_preds, ytest, suffix=" CV Run {}".format(run_id), ax=ax2, tprs=tprs[1])

            result[algorithm[2]].append([np.asarray(train_values),
                                         np.asarray(test_values)])

            final_clf.fit(X, y)

            if not output_proba:
                tpreds = final_clf.predict(Xtest)
                tpreds_train = final_clf.predict(X)
            else:
                tpreds = final_clf.predict_proba(Xtest)[:, 1]
                tpreds_train = final_clf.predict_proba(X)[:, 1]

            predict_result[algorithm[2]].append(np.concatenate([tpreds_train, tpreds]))

        run_id += 1

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=1)
    std = tprs.std(axis=1)

    axes = [ax1, ax2]
    colors = ['b', 'r']

    validation_runs_meanroc.append(mean_tprs[1].tolist()+[auc(base_fpr, mean_tprs[1])])
    testing_runs_meanroc.append(mean_tprs[0].tolist() + [auc(base_fpr, mean_tprs[0])])

    for i, tpr_i in enumerate(mean_tprs):

        tprs_upper = np.minimum(tpr_i + std[i], 1)
        tprs_lower = tpr_i - std[i]

        mean_roc_auc = auc(base_fpr, tpr_i)

        axes[i].plot(base_fpr, tpr_i, colors[i], label='Mean ROC (area={:.3f})'.format(mean_roc_auc))
        axes[i].fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

        axes[i].legend(loc="lower right")

    fig_name = "/local/temporary/clf_vis/{}ROC_AUC_Supervised_{}.png".format(output_prefix, feature_str)

    # fig.tight_layout()
    fig.savefig(fig_name)

    print("\n ************** \n     RESULTS     \n")

    print("ALG \t\t auc \t\t spec \t\t sens ")
    indicator = ['train', 'test']
    for algorithm in algorithms:

        alg_results = result[algorithm[2]]

        for i in range(2):
            ares = [item[i] for item in alg_results]
            mean_perf = np.mean(ares, axis=0)
            sd_perf = np.std(ares, axis=0)

            perf_str = ''
            for m, s in zip(mean_perf, sd_perf):
                perf_str += "{0:0.4f}+-{1:0.4f} \t".format(m, s)

            print("{0} [{2}]\t {1}".format(algorithm[2], perf_str, indicator[i]))

      #  print("[FEATURE PERFORMANCES]")
      #  print(", ".join(map(str, mean_feature_performances.tolist())))

    return predict_result


def cv_eval_xgbboost(X, y, Xtest, ytest, algorithms, rounds=5, feature_str="", output_prefix="", output_proba=False):
    global validation_runs_meanroc
    global testing_runs_meanroc

    y = [y for i, y in enumerate(y) if len(X[i]) == len(X[0])]
    X = [x for x in X if len(x) == len(X[0])]

    X = np.array(X)
    y = np.array(y)

    Xtest = np.array(Xtest)

    assert len(X) == len(y), "Instance and label array do not match... {0} != {1}".format(
        len(X), len(y)
    )

    print("CV Eval classifier: \n ---------- \n Data statistics: \n" +
          "Label: [-1] \t {}\n".format(len(y[y < 0])) +
          "Label: [ 1] \t {}".format(len(y[y > 0])))

    M = MinMaxScaler()
    X = M.fit_transform(X)
    Xtest = M.transform(Xtest)
    stratified_sampler = StratifiedShuffleSplit(n_splits=rounds, test_size=0.2)

    predict_result = {}

    result = {}

    mean_feature_performances = np.zeros(len(X[0]))
    for algorithm in algorithms:
        result.update({algorithm[2]: []})
        predict_result.update({algorithm[2]: []})

    run_id = 0
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 10), dpi=150)
    fig.suptitle("Receiver-operating characteristics [Train set] \n Used features: " + feature_str)
    ax1.set_title('[Train set]')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')

    ax2.set_title('[Test set]')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')

    cv_eval_validation_set = (Xtest, ytest)

    tprs = [[], []]
    final_clf = algorithm[0]
    # pass the parameters to the classifier
    for parameter, value in algorithm[1].items():

        if hasattr(final_clf, parameter):
            # de-capsulate the input parameters (given as list in the params dict)
            setattr(final_clf, parameter, value[0])

    for train_index, test_index in stratified_sampler.split(X, y):

        sw = compute_sample_weight('balanced', y, train_index)

        cv_eval_set = (X[test_index], y[test_index])

        if final_clf.__class__.__name__ is 'XGBClassifier':
            final_clf.fit(X[train_index], y[train_index],
                          eval_set=[cv_eval_set, cv_eval_validation_set],
                          sample_weight=sw,
                          eval_metric=['auc', 'map'])
        else:
            final_clf.fit(X[train_index], y[train_index])

        ppreds = None
        fpredict = getattr(final_clf, "predict_proba", None)
        if fpredict is None:
            fpredict = getattr(final_clf, "decision_function", None)

        if callable(fpredict):
            ppreds = fpredict(X[test_index])

        train_values = auc_analysis(ppreds, y[test_index], suffix=" CV Run {}".format(run_id), ax=ax1, tprs=tprs[0])
        mean_feature_performances = mean_feature_performances + (1.0 / rounds) * final_clf.feature_importances_

        # final clf predict
        tp_preds = None

        fpredict = getattr(final_clf, "predict_proba", None)
        if fpredict is None:
            fpredict = getattr(final_clf, "decision_function", None)

        if callable(fpredict):
            tp_preds = fpredict(Xtest)

        test_values = auc_analysis(tp_preds, ytest, suffix=" CV Run {}".format(run_id), ax=ax2, tprs=tprs[1])

        result[algorithm[2]].append([np.asarray(train_values),
                                     np.asarray(test_values)])

        # final_clf.fit(X, y, eval_set=[cv_eval_validation_set], sample_weight=sw,
        #               eval_metric=['auc', 'map'])

        if not output_proba:
            tpreds = final_clf.predict(Xtest)
            tpreds_train = final_clf.predict(X)
        else:
            tpreds = final_clf.predict_proba(Xtest)[:, 1]
            tpreds_train = final_clf.predict_proba(X)[:, 1]

        predict_result[algorithm[2]].append(tpreds)

        run_id += 1
        final_clf.n_estimators += 15
        final_clf.learning_rate *= 1.0

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=1)
    std = tprs.std(axis=1)

    axes = [ax1, ax2]
    colors = ['b', 'r']

    validation_runs_meanroc.append(mean_tprs[1].tolist() + [auc(base_fpr, mean_tprs[1])])
    testing_runs_meanroc.append(mean_tprs[0].tolist() + [auc(base_fpr, mean_tprs[0])])

    for i, tpr_i in enumerate(mean_tprs):
        tprs_upper = np.minimum(tpr_i + std[i], 1)
        tprs_lower = tpr_i - std[i]

        mean_roc_auc = auc(base_fpr, tpr_i)

        axes[i].plot(base_fpr, tpr_i, colors[i], label='Mean ROC (area={:.3f})'.format(mean_roc_auc))
        axes[i].fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

        axes[i].legend(loc="lower right")

    fig_name = "/local/temporary/clf_vis/{}ROC_AUC_Supervised_{}.png".format(output_prefix, feature_str)

    # fig.tight_layout()
    fig.savefig(fig_name)

    print("\n ************** \n     RESULTS     \n")

    print("ALG \t\t auc \t\t spec \t\t sens ")
    logf = open('/local/temporary/clf_vis/{}_performance.log'.format(output_prefix), 'a')
    indicator = ['train', 'test']
    for algorithm in algorithms:

        alg_results = result[algorithm[2]]

        for i in range(2):
            ares = [item[i] for item in alg_results]
            mean_perf = np.mean(ares, axis=0)
            sd_perf = np.std(ares, axis=0)

            perf_str = ''
            ofile_str = ''
            for m, s in zip(mean_perf, sd_perf):
                perf_str += "{0:0.4f}+-{1:0.4f} \n".format(m, s)
                ofile_str += ";{0:0.4f};{1:0.4f}".format(m, s)

            print("{0} [{2}]\t {1}".format(algorithm[2], perf_str, indicator[i]))
            logf.write("XGB;{};{}{}\n".format(len(y[y < 0]), len(y[y > 0]), ofile_str))

            #  print("[FEATURE PERFORMANCES]")
            #  print(", ".join(map(str, mean_feature_performances.tolist())))

    logf.close()

    return predict_result


def visualize_prediction( algorithms, pred_results, M, labels, train_len=0, output_prefix=""):
    visualize_prediction.counter += 1

    data_counter = 0
    validation_offset = 0
    re_eval_scores = []
    re_eval_labels = []

    for algorithm in algorithms:
        alg_name = algorithm[2]

        mean_preds = np.mean(np.array(pred_results[alg_name]), axis=0)
        max_preds = np.max(np.array(pred_results[alg_name]), axis=0)
        min_preds = np.min(np.array(pred_results[alg_name]), axis=0)
        sd_preds = np.std(np.array(pred_results[alg_name]), axis=0)
        # sd_preds = np.std(np.array(pred_results[alg_name])-np.array(labels), axis=0)

        scalars = mean_preds * 100.

        border_scalars = np.zeros_like(scalars)
        for li, label in enumerate(labels):
            if label > 0:
                border_scalars[li] = 100 * min_preds[li]
            else:
                border_scalars[li] = 100 * max_preds[li]

        for m in M:
            metadata = m

            image_name = os.path.basename(metadata[0])
            patch_size = 1024 # int(os.path.basename(metadata[0]).split('_')[3][1:])
            if '_s512_' in image_name:
                patch_size = 512

            if '_s256_' in image_name:
                patch_size = 256

            image_name_parts = image_name.split('_')
            image_root = "/datagrid/Medical/microscopy/CAMELYON16/training"

            if 'tumor' in metadata[0]:
                image_root += "/tumor/"
                image_path = image_root + "tumor_{}.tif".format(image_name_parts[2])

            elif 'normal' in metadata[0]:
                image_root += "/normal/"
                image_path = image_root + "Normal_{}.tif".format(image_name_parts[2])

            else:
                image_root = "/datagrid/Medical/microscopy/CAMELYON16/testing/"
                idx = 2
                if image_name_parts[1] == 'annot':
                    idx = 3
                image_path = image_root + "Test_{}.tif".format(image_name_parts[idx])

            if 'Test_116' in image_path:
                continue

            si = CamelyonSlideImage(image_path, None)
            # print(" [VisOutput] Loading image {}".format(image_path))
            if not os.path.exists(image_path):
                print (" !! FAILED, IMAGE {} NOT FOUND !!".format(image_path))
                break

            output_map = []

            try:
                patchvis = si.get_patch_visualization(6, metadata[2], patch_size,
                                                      scalars=scalars[validation_offset+m[3]:validation_offset+m[3]+len(m[2])],
                                                      border_scalars=border_scalars[validation_offset+m[3]:validation_offset+m[3]+len(m[2])],
                                                      line_thickness=-1,
                                                      show=False, filled=True, output_map=output_map,
                                                      offset=20, scalar_scale=0.5)
            except ValueError as e:
                print("[[ERROR]] Exception while creating visualization for {} \n Exception: {}".format(image_name, e))
                continue

            re_eval_labels += labels[validation_offset + m[3]:validation_offset + m[3] + len(m[2])]

            if data_counter < train_len:
                cv2.putText(patchvis, 'test', (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(patchvis, 'vali', (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)

            data_counter += 1

            if data_counter == train_len:
                validation_offset = m[3]+len(m[2])

            if 'camgen' not in output_prefix:
                cv2.imwrite("/local/temporary/clf_vis/{}_l0_vis_".format(output_prefix)+image_name+"_"+alg_name+"features_{:d}.png".format(visualize_prediction.counter), patchvis)

            if output_map is not None and len(output_map):
                cv2.imwrite("/local/temporary/clf_vis/{}_l0_vis_".format(output_prefix)+image_name+"_"+alg_name+"predictionscores_{:d}.png".format(visualize_prediction.counter), output_map[0])

                re_eval_scores += output_map[1]

                positive_indices = []
                negative_indices = []

                current_scores = np.array(output_map[1])

                output_k_patches = 5
                for p_i, patch in enumerate(metadata[2]):
                    patch_pos = patch[0]
                    patch_label = patch[1]

                    if patch_label == 'NO':
                        negative_indices.append(p_i)
                    if patch_label == 'TU':
                        positive_indices.append(p_i)

                image_id, ext = os.path.splitext(os.path.basename(image_path))
                output_path_base = '/local/temporary/clf_vis/' + output_prefix[:-5] + image_id
                output_list = 'file;patch_idx;label;loc_x;loc_y;size;pred_score\n'
                if len(negative_indices):
                    try:
                        negative_scores = current_scores[np.asarray(negative_indices)]
                        true_negative = np.argpartition(negative_scores, output_k_patches)[:output_k_patches]
                        false_positive = np.argpartition(negative_scores, -output_k_patches)[-output_k_patches:]

                        if not os.path.exists(output_path_base):
                            os.mkdir(output_path_base)

                        for tn_idx in true_negative:
                            patch = si.load_patch((metadata[2][tn_idx])[0], size=patch_size, level=0)

                            cv2.imwrite(output_path_base+'/TrueNeg_{}_{}.png'.format(tn_idx,
                                                                                     2*int(negative_scores[tn_idx]-20)),
                                        patch)
                            output_list += '{};{};NO;{};{};{};{}\n'.format(image_id,
                                                                           tn_idx,
                                                                           ((metadata[2][tn_idx])[0])[0],
                                                                           ((metadata[2][tn_idx])[0])[1],
                                                                           patch_size,
                                                                           1 / 50. * (negative_scores[tn_idx] - 20)
                                                                           )

                        for fp_idx in false_positive:
                            patch = si.load_patch((metadata[2][fp_idx])[0], size=patch_size, level=0)

                            cv2.imwrite(
                                output_path_base + '/FalsePos_{}_{}.png'.format(fp_idx,
                                                                                2 * int(negative_scores[fp_idx] - 20)),
                                patch)

                            output_list += '{};{};NO;{};{};{};{}\n'.format(image_id,
                                                                           fp_idx,
                                                                           ((metadata[2][fp_idx])[0])[0],
                                                                           ((metadata[2][fp_idx])[0])[1],
                                                                           patch_size,
                                                                           1 / 50. * (negative_scores[fp_idx] - 20)
                                                                           )

                        with open(output_path_base+'/exported_patch_list.csv', 'a') as f:
                            f.write(output_list)

                        f.close()

                    except ValueError as e:
                        print(" ||EXCEPTION|| Skip neg-vis output for {}, \n got exception {}".format(image_name, e))
                        continue

                if len(positive_indices):
                    try:
                        positive_scores = current_scores[np.asarray(positive_indices)]
                        false_negative = np.argpartition(positive_scores, output_k_patches)[:output_k_patches]
                        true_positive = np.argpartition(positive_scores, -output_k_patches)[-output_k_patches:]

                        if not os.path.exists(output_path_base):
                            os.mkdir(output_path_base)

                        for tp_idx in true_positive:
                            patch = si.load_patch((metadata[2][tp_idx])[0], size=patch_size, level=0)

                            cv2.imwrite(output_path_base+'/TruePos_{}_{}.png'.format(tp_idx,
                                                                                     2 * int(positive_scores[tp_idx]-20)),
                                        patch)

                            output_list += '{};{};TU;{};{};{};{}\n'.format(image_id,
                                                                           tp_idx,
                                                                           ((metadata[2][tp_idx])[0])[0],
                                                                           ((metadata[2][tp_idx])[0])[1],
                                                                           patch_size,
                                                                           1 / 50. * (positive_scores[tp_idx] - 20)
                                                                           )

                        for fn_idx in false_negative:
                            patch = si.load_patch((metadata[2][fn_idx])[0], size=patch_size, level=0)

                            cv2.imwrite(
                                output_path_base + '/FalseNeg_{}_{}.png'.format(fn_idx,
                                                                                2*int(positive_scores[fn_idx]-20)),
                                patch)

                            output_list += '{};{};TU;{};{};{};{}\n'.format(image_id,
                                                                           fn_idx,
                                                                           ((metadata[2][fn_idx])[0])[0],
                                                                           ((metadata[2][fn_idx])[0])[1],
                                                                           patch_size,
                                                                           1 / 50. * (positive_scores[fn_idx] - 20)
                                                                           )

                        with open(output_path_base + '/exported_patch_list.csv', 'a') as f:
                            f.write(output_list)

                        f.close()

                    except ValueError as e:
                        print(" ||EXCEPTION|| Skip pos-vis output for {}, \n got exception {}".format(image_name, e))
                        continue

    if len(re_eval_scores):
        roc_auc, spec, sens = auc_analysis(np.array(re_eval_scores[len(M[0][2]):]),
                                           np.array(re_eval_labels[len(M[0][2]):]))

        print("Re-evaluated AUC (over min prediction scores in patch) \n ================== \n "
              "ROC-AUC {} \t Spec {} \t Sens {}".format(roc_auc, spec, sens)
              )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Supervised, path-based classification")
    parser.add_argument('-r', '--study_root', type=str, default="/datagrid/personal/herinjan/milearning/microscopy-python/feature_descriptors/",
                        help="Root directory with feature descriptors (.csv file per one WSI)", required=False)
    parser.add_argument('-no', '--normal', type=str, default="*cam16_normal4_0[0-5]*l6_s512_mcontext012_bgcorr_mswvlt_hehist_fixoptsep_mscale_features.csv",
                        required=False, help='Parse pattern for normal (healthy) training data')
    parser.add_argument('-tu', '--tumor', type=str, default="*cam16_tumor3_0[0-5]*l6_s512_mcontext012_bgcorr_mswvlt_hehist_fixoptsep_mscale_features.csv",
                        required=False, help='Parse pattern for tumor training data')
    parser.add_argument('-te', '--testing', type=str, default="*cam16_annot_testing3_0[0-5]*__l6_s512_mcontext012_bgcorr_mswvlt_hehist_fixoptsep_mscale_features.csv",
                        required=False, help='Parse pattern for testing (mixed labels) training data')
    parser.add_argument('-o', '--output_prefix', type=str, default="cam16_first_stage_out/cc_test2_",
                        required=False, help='Output prefix for result images')
    parser.add_argument('-sf', '--study_features', type=str, nargs='+', required=True,
                        help='Classification features to be used (specified by a prefix)')
    parser.add_argument('-c', '--classifier', type=str, default="XGB",
                        required=False, help='Classifier type')
    parser.add_argument('-n', '--n_repetitions', type=int, default=5,
                        required=False, help='Number of repetitions')
    parser.add_argument('--n_normal', type=int, help='Number (max) of normal patches extracted per slide image', default=1000)
    parser.add_argument('--n_jobs', type=int, help='Number (max) jobs (threads) in XGBoost Classifier',
                        default=8)
    parser.add_argument('--title', type=str, default='', required=False, help="Title string for ROC-AUC figure")

    args = parser.parse_args()

    # Parsed arguments
    study_root = args.study_root
    test_root = args.study_root
    tumor_parse_str = args.tumor
    normal_parse_str = args.normal
    test_parse_str = args.testing
    clf = args.classifier
    output_prefix = args.output_prefix
    title_string = args.title

    # Fixed arguments
    study_groups = ['']
    study_labels = [-1, 1]
    group_dict = dict(zip(study_groups, study_labels))

    basic_features = []  # do not include patch location

    lin_svm_params = {"C": [1.0, 10.0, 100.0]}
    rf_params = {"n_estimators": [50],
                 "max_depth": [8],
                 "min_samples_leaf": [5],
                 "min_samples_split": [25],
                 "max_features": [0.25]}
    xgb_params = {"n_estimators": [60],
                  "max_depth": [5],
                  "learning_rate": [0.2],
                  "n_jobs": [args.n_jobs],
                  "silent": [True],
                  "subsample": [0.8],
                  "reg_lambda": [3],
                  "reg_alpha": [1],
                  "colsample_bytree": [0.8],
                  "min_child_weight": [25],
                  "max_delta_step": [1],
                  "gamma": [5]}
    knn_params = {"n_neighbors": [3, 5, 10, 40, 60]}
    label_dict = {'NO': -1, 'TU': 1, 'BO': 1, 'OT': 1}

    if clf == 'GB':
        algorithms = [#(LinearSVC(), lin_svm_params, "Linear SVM"),
                      #(RandomForestClassifier(class_weight="balanced", n_jobs=8,
                      #                        bootstrap=True, criterion='gini'), rf_params, "RandomForest"),
                      #(GradientBoostingClassifier(loss='deviance', init=init_algo), rf_params, "GradientBoosting")
                      (GradientBoostingClassifier(verbose=2, warm_start=True, subsample=0.8), rf_params, "GradientBoosting")
                      # (KNeighborsClassifier(), knn_params, "KNN Clf")
                      ]
    elif clf == 'XGB':
        algorithms = [
            (XGBClassifier(booster='gbtree'), xgb_params, "XGBoosting")
        ]

    else:
        algorithms = [#(LinearSVC(), lin_svm_params, "Linear SVM"),
                      (RandomForestClassifier(class_weight="balanced", n_jobs=9,
                                              bootstrap=True, criterion='gini'), rf_params, "RandomForest"),
                      #(GradientBoostingClassifier(loss='deviance', init=init_algo), rf_params, "GradientBoosting")
                      #(GradientBoostingClassifier(), rf_params, "GradientBoosting")
                      # (KNeighborsClassifier(), knn_params, "KNN Clf")
                      ]

    input_csv_files = sorted(glob(study_root + '/' + study_groups[0] + '/' + normal_parse_str))
    input_csv_files2 = sorted(glob(test_root + '/' + study_groups[0] + '/' + tumor_parse_str))
    test_csv_files = sorted(glob(test_root + '/' + study_groups[0] + '/' + test_parse_str))

    n_repetitions = args.n_repetitions

    train_splits = []
    test_splits = []

    study_features = args.study_features
    print("> All patches, split into -1 (TU) and +1 (NO) \n> Features: " + ", ".join(study_features))
    print("> Validation on unseen datasets, all patches")

    Xn, yn, Mn = csv_io.import_study_from_csv(study_root=study_root, group_dict=group_dict,
                                              skip_list=('OT', 'BO'),
                                              groups=study_groups, features=study_features,
                                              parse_string='', label_dict=label_dict,
                                              file_list=input_csv_files, restrict_size=True,
                                              max_patches=args.n_normal)

    Xt, yt, Mt = csv_io.import_study_from_csv(study_root=study_root, group_dict=group_dict,
                                              skip_list=('OT', 'NO', 'BO'),
                                              groups=study_groups, features=study_features,
                                              parse_string='', label_dict=label_dict,
                                              file_list=input_csv_files2,
                                              restrict_size=False)

    XnII, ynII, MnII = csv_io.import_study_from_csv(study_root=study_root, group_dict=group_dict,
                                                    skip_list=('OT'),
                                                    groups=study_groups, features=study_features,
                                                    parse_string='', label_dict=label_dict,
                                                    file_list=input_csv_files,
                                                    restrict_size=False)

    XtII, ytII, MtII = csv_io.import_study_from_csv(study_root=study_root, group_dict=group_dict,
                                                    skip_list=('OT'),
                                                    groups=study_groups, features=study_features,
                                                    parse_string='', label_dict=label_dict,
                                                    file_list=input_csv_files2,
                                                    restrict_size=False)

    Xvali, yvali, Mvali = csv_io.import_study_from_csv(study_root=test_root, group_dict=group_dict,
                                                       groups=study_groups, features=study_features,
                                                       parse_string=test_parse_str, skip_list=('OT'),
                                                       label_dict=label_dict, file_list=test_csv_files)

    n_split = int(len(Mn) * 0.2)
    t_split = int(len(Mt) * 0.2)

    n_indices = np.arange(0, len(Mn))
    t_indices = np.arange(0, len(Mt))
    for rep_i in range(n_repetitions):

        # get random splits
        np.random.shuffle(n_indices)
        n_train_idx = n_indices[:-n_split]
        n_snd_phase_ids = n_indices[-n_split:]

        np.random.shuffle(t_indices)
        t_train_idx = t_indices[:-t_split]
        t_snd_phase_ids = t_indices[-t_split:]

        print("[Repetition {}] Phase I / Phase II set lengths \n {}  +  {}".format(rep_i, len(n_train_idx)+len(t_train_idx),
                                                                                   n_split + t_split))

        # Compose training/second phase training groups
        Xa = []
        ya = []
        Ma = []

        Xa2 = []
        ya2 = []
        Ma2 = []

        offset = 0
        for n_idx in n_train_idx:
            Mt_entry = Mn[n_idx]
            Xa += Xn[Mt_entry[3]:Mt_entry[3]+len(Mt_entry[2])]
            ya += yn[Mt_entry[3]:Mt_entry[3]+len(Mt_entry[2])]
            Ma.append((Mt_entry[0], Mt_entry[1], Mt_entry[2][:], offset))
            offset += len(Mt_entry[2])

        for t_idx in t_train_idx:

            Mt_entry = Mt[t_idx]
            Xa += Xt[Mt_entry[3]:Mt_entry[3]+len(Mt_entry[2])]
            ya += yt[Mt_entry[3]:Mt_entry[3]+len(Mt_entry[2])]
            Ma.append((Mt_entry[0], Mt_entry[1], Mt_entry[2][:], offset))
            offset += len(Mt_entry[2])

        offset = 0
        for n_idx in n_snd_phase_ids:

            Mt_entry = MnII[n_idx]
            Xa2 += XnII[Mt_entry[3]:Mt_entry[3]+len(Mt_entry[2])]
            ya2 += ynII[Mt_entry[3]:Mt_entry[3]+len(Mt_entry[2])]
            Ma2.append((Mt_entry[0], Mt_entry[1], Mt_entry[2][:], offset))
            offset += len(Mt_entry[2])

        for t_idx in t_snd_phase_ids:
            Mt_entry = MtII[t_idx]
            Xa2 += XtII[Mt_entry[3]:Mt_entry[3]+len(Mt_entry[2])]
            ya2 += ytII[Mt_entry[3]:Mt_entry[3]+len(Mt_entry[2])]
            Ma2.append((Mt_entry[0], Mt_entry[1], Mt_entry[2][:], offset))
            offset += len(Mt_entry[2])

        visualize_prediction.counter = 0

        pred_results = cv_eval_xgbboost(Xa, ya, algorithms=algorithms,
                                        Xtest=Xa2, ytest=ya2,
                                        feature_str="+".join(study_features),
                                        output_prefix=output_prefix+"_rep{}_".format(rep_i),
                                        rounds=5, output_proba=True)

        visualize_prediction(algorithms, pred_results, Ma2, ya2, train_len=len(Ma2),
                             output_prefix=output_prefix+"_rep{}_".format(rep_i))

        # join training sets
        Mfulltrain = Ma
        Xfulltrain = Xa
        yfulltrain = ya
        # Xfulltrain = []
        # yfulltrain = []
        # Mfulltrain = []
        #
        # offset = 0
        # for mi in range(len(Ma)):
        #     Mt_entry = Ma[mi]
        #     Xfulltrain += Xa[Mt_entry[3]:Mt_entry[3]+len(Mt_entry[2])]
        #     yfulltrain += ya[Mt_entry[3]:Mt_entry[3]+len(Mt_entry[2])]
        #     Mfulltrain.append((Mt_entry[0], Mt_entry[1], Mt_entry[2][:], offset))
        #     offset += len(Mt_entry[2])
        #
        # for mi in range(len(Ma2)):
        #     Mt_entry = Ma2[mi]
        #     Xfulltrain += Xa2[Mt_entry[3]:Mt_entry[3]+len(Mt_entry[2])]
        #     yfulltrain += ya2[Mt_entry[3]:Mt_entry[3]+len(Mt_entry[2])]
        #     Mfulltrain.append((Mt_entry[0], Mt_entry[1], Mt_entry[2][:], offset))
        #     offset += len(Mt_entry[2])

        pred_results = cv_eval_xgbboost(Xfulltrain, yfulltrain, algorithms=algorithms,
                                        Xtest=Xvali, ytest=yvali,
                                        feature_str="+".join(study_features),
                                        output_prefix=output_prefix+"_fulltrain_rep{}_".format(rep_i),
                                        rounds=5, output_proba=True)

        visualize_prediction(algorithms, pred_results, Mvali, yvali, train_len=len(Mvali),
                             output_prefix=output_prefix+"_fulltrain_rep{}_".format(rep_i))

    # visualize mean roc curves
    for j in range(0, 2):

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 10), dpi=150)
        fig.suptitle("Receiver-operating characteristics [Over different images splits] \n "
                     "Classifier: RF, Features/Preprocessing: {}".format(title_string))
        ax1.set_title('[Train-validation set]')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')

        ax2.set_title('[Validation set]')
        if j> 0:
            ax2.set_title('[Test set]')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')

        for i in range(n_repetitions):
            ax2.plot(base_fpr, validation_runs_meanroc[2*i+j][:-1], label='Mean ROC (area={:.3f})'.format(
                validation_runs_meanroc[i][-1]))
            ax1.plot(base_fpr, testing_runs_meanroc[2*i+j][:-1], label='Mean ROC (area={:.3f})'.format(
                testing_runs_meanroc[i][-1]))

        ax1.legend(loc="lower right")
        ax2.legend(loc="lower right")

        fig_name = "/local/temporary/clf_vis/{}ROC_AUC_Supervised_Repetitions_{:02d}.png".format(output_prefix, j)
        # fig.tight_layout()
        fig.savefig(fig_name)

