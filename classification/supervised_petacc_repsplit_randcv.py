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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, f1_score
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_randf
from sklearn.model_selection import RandomizedSearchCV

base_fpr = np.linspace(0, 1, 101)

validation_runs_meanroc = []
testing_runs_meanroc = []


def _clf_score(clf, gt, preds):

    oob_score = getattr(clf, 'oob_score_', None)
    #if oob_score is not None:
    #    return oob_score
    #else:
    #    return f1_score(gt, preds)
    return f1_score(gt, preds)

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

    rs = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, scoring='f1_weighted',
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

    gs = GridSearchCV(estimator=clf,
                      param_grid=params, scoring='accuracy', error_score=0)
    gs.fit(X=X[train_idx], y=y[train_idx])

    print("GridSearch Result")
    print(gs.best_params_)

    final_clf = gs.best_estimator_

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


def cv_eval_classifier(X, y, Xtest, ytest, algorithms, rounds=10, feature_str="", output_prefix=""):


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
                                                                X=X, y=y, train_idx=train_index, test_idx=test_index)
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

            tpreds = final_clf.predict(Xtest)
            predict_result[algorithm[2]].append(tpreds)

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

    # fig_name = "/tmp/{}ROC_AUC_Supervised_{}.png".format(output_prefix, feature_str)

    # fig.tight_layout()
    # fig.savefig(fig_name)

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


def visualize_prediction( algorithms, pred_results, M, labels, output_prefix=""):
    visualize_prediction.counter += 1

    for m in M:
        metadata = m

        for algorithm in algorithms:
            alg_name = algorithm[2]

            mean_preds = np.sum(np.array(pred_results[alg_name])-np.array(labels), axis=0)
            sd_preds = np.std(np.array(pred_results[alg_name])-np.array(labels), axis=0)

            scalars = np.abs(mean_preds * 10. / np.max(mean_preds))

            petacc_root = "/datagrid/Medical/microscopy/petacc3"
            image_name = os.path.basename(metadata[0])[:8]
            patch_size = int(os.path.basename(metadata[0]).split('_')[3][1:])
            image_path = petacc_root + "/batch_2/" + image_name + ".ndpi"
            if not os.path.exists(image_path):
                image_path = petacc_root + "/batch_1/" + image_name + ".ndpi"

            si = NDPISlideImage(image_path, None)
            patchvis = si.get_patch_visualization(5, metadata[2], patch_size,
                                                  scalars=scalars[m[3]:m[3]+len(m[2])],
                                                  line_thickness=2,
                                                  show=False )

            cv2.imwrite("/tmp/{}_l0_vis_".format(output_prefix)+image_name+"_"+alg_name+"features_{:d}.png".format(visualize_prediction.counter), patchvis)


if __name__ == "__main__":

    study_root = "/datagrid/personal/herinjan/milearning/microscopy-python/features/training"
    study_groups = ["petacc"]
    test_root = "/datagrid/personal/herinjan/milearning/microscopy-python/features/validation"
    study_labels = [-1, 1]
    group_dict = dict(zip(study_groups, study_labels))
    parse_str = "E*s1024_pext_0_stru_his_txt_features.csv"
    test_parse_str = "*s1024_pext_0_stru_his_txt_features.csv"

    output_prefix="Supervised_Classif"
    if len(sys.argv) > 1:
        parse_str = sys.argv[1]         # "[B]*s1024_pext_0_stru_his_txt_features.csv"
        test_parse_str = sys.argv[2]    # "*s1024_pext_0_stru_his_txt_features.csv"
        output_prefix = sys.argv[3]

    basic_features = []  # do not include patch location

    lin_svm_params = {"C": [1.0, 10.0, 100.0]}
    rf_params = {"n_estimators": [10, 30, 50, 100],
                 "max_depth": [10, 20, 40, None],
                 "min_samples_leaf": [5, 20, 50, 100],
                 "max_features": [0.2, 0.5, 'auto']}

    rf_param_dist = {"n_estimators": sp_randint(40, 100),
                     "max_depth": sp_randint(10, 25),
                     "min_samples_leaf": sp_randint(1, 10),
                     "min_samples_split": sp_randint(4, 15),
                     "max_features": sp_randf(loc=0.1, scale=0.45)
                     }
    knn_params = {"n_neighbors": [3, 5, 10, 40, 60]}
    label_dict = {'NO': -1, 'TU': 1, 'BO': 1, 'OT': 1}

    algorithms = [#(LinearSVC(), lin_svm_params, "Linear SVM"),
                  (RandomForestClassifier(class_weight="balanced", oob_score=False, n_jobs=4,
                                          bootstrap=True, criterion='gini'), rf_param_dist, "RandomForest", "RAND"),
                  # (KNeighborsClassifier(), knn_params, "KNN Clf")
                  ]

    add_features = [#["sprel_", "sp_"],
                    #  ["tx_", "wx_"],
                    # ["wx_cE", "wx_cH", "sprel_", "sp_"],
                    # ["HIST_", "wx_cE", "wx_cH"],
                    ["hist_u", "hist_cG", "wx_", "sp_num", "sp_mean", "rel_area", "scale"],
                    # ["tx_", "wx_", "sp_", "sprel_"]
                    ]
                     #["sp_"], ["sp_", "sprel_"], ["sp_", "HIST_"],
                     #["wx_"], ["tx_"], ["tx_", "wx_"], ["tx_", "wx_", "greyco_"],
                     #["HIST_", "wx_"], ["HIST_", "tx_"], ["HIST_", "tx_", "wx_"], ["HIST_", "tx_", "wx_", "greyco_"]]

    input_csv_files = sorted(glob(study_root + '/' + study_groups[0] + '/' + parse_str))
    #print("Found {} files matching the parse string".format(len(input_csv_files)))
    input_csv_files += sorted(glob(test_root + '/' + study_groups[0] + '/' + test_parse_str))

    #print("Found TOTAL {} files matching the parse string".format(len(input_csv_files)))
    
    n_repetitions = 5
    np.random.seed(0)

    train_splits = []
    test_splits = []

    for rep_i in range(n_repetitions):

        np.random.shuffle(input_csv_files)
        split = int(0.15 * len(input_csv_files))

        train_splits.append( input_csv_files[:] )
        test_splits.append( input_csv_files[-split:] )

    for rep_i in range(n_repetitions):

        train_csv_files = train_splits[rep_i]
        test_csv_files = test_splits[rep_i]

        print("[Repetition {}] Test/Validation set lengths \n {}  +  {}".format(rep_i, len(train_csv_files), len(test_csv_files)))

        visualize_prediction.counter = 100
        for feature in add_features:
            study_features = basic_features + feature

            print("> All patches, split into -1 (TU) and +1 (NO) \n> Features: "+", ".join(study_features))
            print("> Validation on unseen datasets, all patches")

            Xa, ya, Ma = csv_io.import_study_from_csv(study_root=study_root, group_dict=group_dict,
                                                      groups=study_groups, features=study_features,
                                                      parse_string=parse_str, label_dict=label_dict, file_list=train_csv_files)

            Xtest, ytest, Mtest = csv_io.import_study_from_csv(study_root=test_root, group_dict=group_dict,
                                                               groups=study_groups, features=study_features,
                                                               parse_string=test_parse_str, skip_list=(),
                                                               label_dict=label_dict, file_list=test_csv_files)

            pred_results = cv_eval_classifier(Xa, ya, algorithms=algorithms,
                                              Xtest=Xtest, ytest=ytest,
                                              feature_str="+".join(study_features),
                                              output_prefix=output_prefix)

       # visualize_prediction(algorithms, pred_results, Mtest, ytest, output_prefix=output_prefix)

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

    fig_name = "/tmp/{}ROC_AUC_Supervised_Repetitions.png".format(output_prefix)
    # fig.tight_layout()
    fig.savefig(fig_name)

