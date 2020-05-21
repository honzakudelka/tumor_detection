import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

from classification import csv_io
import csv
import numpy as np
import cv2
import sys

from glob import glob
from microscopyio.slide_image import NDPISlideImage

# classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, f1_score
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pickle

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

    with open('/tmp/petacc_bestparams.log', 'a') as fh:
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
    roc_idx = np.argmin(selection)

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
            current_clf = algorithm[0]

            if hasattr(current_clf, 'estimators_') and len(current_clf.estimators_) > 0:
                current_clf.estimators_ = []

            preds, ppreds, final_clf = _hyperparam_opt(clf=current_clf, params=algorithm[1],
                                                       X=X, y=y, train_idx=train_index, validation_idx=test_index)

            train_values = auc_analysis(ppreds, y[test_index], suffix=" CV Run {}".format(run_id), ax=ax1, tprs=tprs[0])
            mean_feature_performances = mean_feature_performances + (1.0/rounds) * final_clf.feature_importances_

            # final clf predict (first fit on all data)
            # if isinstance(final_clf, GradientBoostingClassifier):
            #     final_clf.n_estimators += 15
            #     final_clf.warm_start = True

            final_clf.fit(X, y)

            with open('/local/temporary/clf_output/grad_boosting_out.log', 'a') as gbf:
                if hasattr(final_clf, 'train_score_'):
                    gbf.write('TrainLoss;{};'.format(run_id)+';'.join(map(str, final_clf.train_score_))+'\n')
                if hasattr(final_clf, 'oob_improvement_'):
                    gbf.write('OOBImprov;{};'.format(run_id) + ';'.join(map(str, final_clf.oob_improvement_))+'\n')

                gbf.close()

            tp_preds = None

            fpredict = getattr(final_clf, "predict_proba", None)
            if fpredict is None:
                fpredict = getattr(final_clf, "decision_function", None)

            if callable(fpredict):
                tp_preds = fpredict(Xtest)

            test_values = auc_analysis(tp_preds, ytest, suffix=" CV Run {}".format(run_id), ax=ax2, tprs=tprs[1])

            result[algorithm[2]].append([np.asarray(train_values),
                                         np.asarray(test_values)])

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


def visualize_prediction( algorithms, pred_results, M, labels, train_len=0, output_prefix=""):
    visualize_prediction.counter += 1

    data_counter = 0
    validation_offset = 0
    imtype = ''
    for m in M:
        metadata = m

        for algorithm in algorithms:
            alg_name = algorithm[2]

            # mean_preds = np.sum(np.array(pred_results[alg_name])-np.array(labels), axis=0)
            mean_preds = np.sum(np.array(pred_results[alg_name]), axis=0)
            max_preds = np.max(np.array(pred_results[alg_name]), axis=0)
            min_preds = np.min(np.array(pred_results[alg_name]), axis=0)
            sd_preds = np.sum(np.array(pred_results[alg_name]), axis=0)
            # sd_preds = np.std(np.array(pred_results[alg_name])-np.array(labels), axis=0)

            scalars = np.abs(mean_preds * 100. / np.max(mean_preds))

            border_scalars = np.zeros_like(scalars)
            for li, label in enumerate(labels):
                if label > 0:
                    border_scalars[li] = 100 * min_preds[li]
                else:
                    border_scalars[li] = 100 * max_preds[li]

            petacc_root = "/datagrid/Medical/microscopy/petacc3"
            image_name = os.path.basename(metadata[0])[:8]
            patch_size = int(os.path.basename(metadata[0]).split('_')[3][1:])
            image_path = petacc_root + "/batch_2/" + image_name + ".ndpi"
            if not os.path.exists(image_path):
                image_path = petacc_root + "/batch_1/" + image_name + ".ndpi"

            si = NDPISlideImage(image_path, None)
            patchvis = si.get_patch_visualization(6, metadata[2], patch_size,
                                                  scalars=scalars[validation_offset+m[3]:validation_offset+m[3]+len(m[2])],
                                                  border_scalars=border_scalars[validation_offset+m[3]:validation_offset+m[3]+len(m[2])],
                                                  line_thickness=1,
                                                  show=False, filled=True)

            if data_counter < train_len:
                cv2.putText(patchvis, 'test', (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
                imtype='test'
            else:
                cv2.putText(patchvis, 'vali', (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
                imtype='vali'

            data_counter += 1

            if data_counter == train_len:
                validation_offset = m[3]+len(m[2])

            cv2.imwrite("/local/temporary/clf_vis/{}".format(output_prefix)+image_name+"_{}_".format(imtype)+alg_name+"features_{:d}.png".format(visualize_prediction.counter), patchvis)


def get_input_files_multiscale( candidate_list, suffix_list):
    input_csv_files = []

    for file in candidate_list:
        current_file_list = [file]
        all_available = True
        for suffix in suffix_list:
            parse_cand = os.path.dirname(file) + '/' + os.path.basename(file)[:8] + suffix
            candidate = glob(parse_cand)
            if len(candidate) < 1:
                all_available = False
                print("[WARNING] No file found for {}".format(parse_cand))
            else:
                current_file_list.append(candidate[0])

        if all_available:
            input_csv_files.append(current_file_list)

    return input_csv_files


if __name__ == "__main__":

    study_root = "/datagrid/Medical/microscopy/petacc3/patches_metadata/feature_descriptors"
    study_groups = [""]
    test_root = "/datagrid/Medical/microscopy/petacc3/patches_metadata/feature_descriptors"
    study_labels = [-1, 1]
    group_dict = dict(zip(study_groups, study_labels))
    parse_str = "E*s1024_pext_0_stru_his_txt_features.csv"
    test_parse_str = "*s1024_pext_0_stru_his_txt_features.csv"

    clf = 'RF'

    output_prefix="Supervised_Classif"
    if len(sys.argv) > 1:
        parse_str = sys.argv[1]         # "[B]*s1024_pext_0_stru_his_txt_features.csv"
        test_parse_str = sys.argv[2]    # "*s1024_pext_0_stru_his_txt_features.csv"
        output_prefix = sys.argv[3]

    if len(sys.argv) > 5:
        clf = sys.argv[5]

    multiscale = False
    if ';' in parse_str or ';' in test_parse_str:
        multiscale = True
        suffix_list = str(parse_str).split(';')
        test_suffix_list = str(test_parse_str).split(';')

        if len(suffix_list) < len(test_suffix_list):
            suffix_list = test_suffix_list

    basic_features = []  # do not include patch location

    lin_svm_params = {"C": [1.0, 10.0, 100.0]}
    # rf_params = {"n_estimators": [90],
    #              "max_depth": [14],
    #              "min_samples_leaf": [3],
    #              "min_samples_split": [8],
    #              "max_features": [0.37]}
    rf_params = {"n_estimators": [60],
                 "max_depth": [15],
                 "min_samples_leaf": [4],
                 "min_samples_split": [11],
                 "max_features": [0.23]}
    xgb_params = {"n_estimators": [200],
                  "max_depth": [50],
                  "n_jobs": [8],
                  "silent": [True]}
    knn_params = {"n_neighbors": [3, 5, 10, 40, 60]}
    label_dict = {'NO': -1, 'TU': 1, 'BO': 1, 'OT': 1}

    class BoostingInitEstimator(object):

        def __init__(self, rf_clf):
            self.rf_clf = rf_clf

        def fit(self, X, y, sample_weight=None):
            self.rf_clf.fit(X, y, sample_weight)

        def predict(self, X):

            x_preds = np.zeros((len(X), 1))
            x_preds[:, 0] = self.rf_clf.predict(X)[:]
            return x_preds

    gb_init_rf = RandomForestClassifier(class_weight="balanced", n_jobs=8, n_estimators=85, max_depth=22, min_samples_leaf=3,
                                        min_samples_split=10, max_features=0.3)

    init_algo = BoostingInitEstimator(gb_init_rf)

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
            (XGBClassifier(), xgb_params, "XGBoosting")
        ]
    else:
        algorithms = [#(LinearSVC(), lin_svm_params, "Linear SVM"),
                      (RandomForestClassifier(class_weight="balanced", n_jobs=9,
                                              bootstrap=True, criterion='gini'), rf_params, "RandomForest"),
                      #(GradientBoostingClassifier(loss='deviance', init=init_algo), rf_params, "GradientBoosting")
                      #(GradientBoostingClassifier(), rf_params, "GradientBoosting")
                      # (KNeighborsClassifier(), knn_params, "KNN Clf")
                      ]

    add_features = [# ["sprel_", "sp_"],
                    #  ["tx_", "wx_"],
                    #["wx_cE", "wx_cH", "sprel_", "sp_"],
                    # ["HIST_", "wx_cE", "wx_cH"],
                    #["hist_u", "hist_cG", "wx_h", "sp_num", "he_stru", "scale"],
                    #
                    # [#"wx_res_L0", "wx_res_L2",
                    #  #   "wx_res_",
                    #     "L0_he_str", "L2_he_str",
                    #     "L0_sp", "L2_sp", "L4d_hist_c"
                    # ]
        [
            "wx_res_L0", "wx_res_L2", "wx_res_L4",
            "wx_his_L0", "wx_his_L2",
            "L0_he_str", "L2_he_str", "L4_he_str",
            "L0_sp", "L2_sp", "L4_sp"]
                    # ["sp_num"]
                    ]
                     #["sp_"], ["sp_", "sprel_"], ["sp_", "HIST_"],
                     #["wx_"], ["tx_"], ["tx_", "wx_"], ["tx_", "wx_", "greyco_"],
                     #["HIST_", "wx_"], ["HIST_", "tx_"], ["HIST_", "tx_", "wx_"], ["HIST_", "tx_", "wx_", "greyco_"]]


    train_suffix = parse_str
    test_suffix = test_parse_str
    if multiscale:
        train_suffix = suffix_list[0]
        test_suffix = test_suffix_list[0]

    input_csv_files = sorted(glob(study_root + '/' + study_groups[0] + '/' + train_suffix))
    input_csv_files += sorted(glob(test_root + '/' + study_groups[0] + '/' + test_suffix))

    if multiscale:
        input_csv_files = get_input_files_multiscale(input_csv_files, suffix_list[1:])

    n_repetitions = 3
    np.random.seed(5843)

    train_splits = []
    test_splits = []

    for rep_i in range(n_repetitions):

        np.random.shuffle(input_csv_files)
        split = int(0.1 * len(input_csv_files))

        train_splits.append(input_csv_files[:-split])
        test_splits.append(input_csv_files[-split:])

    for rep_i in range(n_repetitions):

        train_csv_files = train_splits[rep_i]
        test_csv_files = test_splits[rep_i]

        print("[Repetition {}] Test/Validation set lengths \n {}  +  {}".format(rep_i, len(input_csv_files)-split, split))

        visualize_prediction.counter = 0
        for feature in add_features:
            study_features = basic_features + feature

            print("> All patches, split into -1 (TU) and +1 (NO) \n> Features: "+", ".join(study_features))
            print("> Validation on unseen datasets, all patches")

            ms_str = ''
            if not multiscale:
                Xa, ya, Ma = csv_io.import_study_from_csv(study_root=study_root, group_dict=group_dict,
                                                      groups=study_groups, features=study_features,
                                                      parse_string=parse_str, label_dict=label_dict, file_list=train_csv_files)

                Xtest, ytest, Mtest = csv_io.import_study_from_csv(study_root=test_root, group_dict=group_dict,
                                                               groups=study_groups, features=study_features,
                                                               parse_string=test_parse_str, skip_list=(),
                                                               label_dict=label_dict, file_list=test_csv_files)
            else:
                Xa, ya, Ma = csv_io.import_study_multiple_per_subject(features=study_features,
                                                                      label_dict=label_dict,
                                                                      file_list=train_csv_files, n_merge=len(input_csv_files[0]))
                Xtest, ytest, Mtest = csv_io.import_study_multiple_per_subject(features=study_features,
                                                                        label_dict=label_dict,
                                                                        file_list=test_csv_files, n_merge=len(input_csv_files[0]))

                ms_str='_msc'

            pred_results = cv_eval_classifier(Xa, ya, algorithms=algorithms,
                                              Xtest=Xtest, ytest=ytest,
                                              feature_str="+".join(study_features),
                                              output_prefix=output_prefix+"{}_rep{}_".format(ms_str, rep_i),
                                              rounds=5, output_proba=True)

            # pred_results = {}
            # pred_results.update({"GradientBoosting": []})
            # for i in range(4):
            #     pred_results['GradientBoosting'].append( 0.5 + 0.5 * np.concatenate((ya, ytest)))

            visualize_prediction(algorithms, pred_results, Ma[-1:]+Mtest, ya+ytest, train_len=1,
                                 output_prefix=output_prefix+"_rep{}_".format(rep_i))
    title_string = " default"
    if len(sys.argv) > 4:
        title_string = sys.argv[4]

    # visualize mean roc curves
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 10), dpi=150)
    fig.suptitle("Receiver-operating characteristics [Over different Training/Validation images splits] \n "
                 "Classifier: RF, Features/Preprocessing: {}".format(title_string))
    ax1.set_title('[Train-test set]')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
 
    ax2.set_title('[Validation set]')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')

    for i in range(n_repetitions):
        ax2.plot(base_fpr, validation_runs_meanroc[i][:-1], label='Mean ROC (area={:.3f})'.format(validation_runs_meanroc[i][-1])) 
        ax1.plot(base_fpr, testing_runs_meanroc[i][:-1], label='Mean ROC (area={:.3f})'.format(testing_runs_meanroc[i][-1]))

    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")

    fig_name = "/local/temporary/clf_vis/{}ROC_AUC_Supervised_Repetitions.png".format(output_prefix)
    # fig.tight_layout()
    fig.savefig(fig_name)

