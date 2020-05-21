import os
import sys
from glob import glob
from scipy import ndimage

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

import cv2
import numpy as np
from skimage.draw import ellipse
from skimage.transform import rotate
from skimage.util import random_noise
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler


def disk_structure(n):
    struct = np.zeros((2 * n + 1, 2 * n + 1))
    x, y = np.indices((2 * n + 1, 2 * n + 1))
    mask = (x - n) ** 2 + (y - n) ** 2 <= n ** 2
    struct[mask] = 1
    return struct.astype(np.bool)


def compute_gaussian_filterings(img_array, sigma_array, thr_array):
    X = []
    counter = 1
    num_img = len(img_array)
    for img in img_array:
        print(" -- Features img {}/{}".format(counter, num_img))
        counter += 1
        features = []

        tot_area = np.count_nonzero(img) + 1

        for thr in thr_array:
            ft_img = (img > thr)

            ft_orig_img = np.multiply(img, ft_img)
            ft_img = ft_img.astype(np.float32)

            cc_stats = cv2.connectedComponentsWithStats(ft_img.astype(np.uint8), 4)

            num_labels = cc_stats[0]
            stats = cc_stats[2]

            cc_areas = np.array([0] + [stats[li, cv2.CC_STAT_AREA] for li in range(1, num_labels)])
            max_area = np.max(cc_areas)
            mean_area = np.mean(cc_areas)

            #features += [100 * max_area / tot_area, 100 * mean_area / tot_area, num_labels - 1]
            features += [max_area, mean_area, num_labels - 1]

            # w_name_str = "Pyramid image at thr {:.2f}".format(thr)
            # cv2.namedWindow(w_name_str, cv2.WINDOW_NORMAL)
            # cv2.resizeWindow(w_name_str, 600, 600)
            # cv2.imshow(w_name_str, np.hstack([img, ft_img, cc_stats[1]]))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            for si, sigma in enumerate(sigma_array):
                ftt_img = cv2.GaussianBlur(ft_orig_img, (0, 0), sigma)

                features.append(np.max(ftt_img))
                #features.append(np.sum(ftt_img))

        X.append(np.array(features))

    return X


def compute_filterings_of_gaussians(img_array, sigma_array, thr_array):
    X = []
    counter = 1
    num_img = len(img_array)
    for img in img_array:
        print(" -- Features img {}/{}".format(counter, num_img))
        counter += 1
        features = []

        for si, sigma in enumerate(sigma_array):
            ft_img = cv2.GaussianBlur(img, (0, 0), sigma)

            for thr in thr_array:
                ftt_img = (ft_img > thr).astype(np.float32)

                # w_name_str = "Pyramid image at thr {:.2f} sigma at {:.2f}".format(thr, sigma)
                # cv2.namedWindow(w_name_str, cv2.WINDOW_NORMAL)
                # cv2.resizeWindow(w_name_str, 600, 600)
                # cv2.imshow(w_name_str, np.hstack([img, ft_img]))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                features.append(np.sum(ftt_img))

            features.append(np.max(ft_img))

        X.append(np.array(features))

    return X


def compute_granulometry_features(img_array, sigma_array, thr_list, gr_sizes=None):
    X = []
    counter = 1
    num_img = len(img_array)
    for img in img_array:
        print(" -- Features img {}/{}".format(counter, num_img))
        counter += 1
        gr_features = []
        for si, sigma in enumerate(sigma_array):
            ft_img = cv2.GaussianBlur(img, (0, 0), sigma)

            # w_name_str = "Pyramid image"
            # cv2.namedWindow(w_name_str, cv2.WINDOW_NORMAL)
            # cv2.resizeWindow(w_name_str, 600, 600)
            # cv2.imshow(w_name_str, ft_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            for thr in thr_list:
                gr_features += [ndimage.binary_opening(ft_img > thr,
                                                       structure=disk_structure(g_size)).sum() for g_size in gr_sizes]

        X.append(np.array(gr_features))

        # print(gr_features)

    return X


def generate_images():
    base_size = (256, 256)
    base_count = 80

    images = []
    labels = []

    for i in range(base_count):
        img = np.zeros(base_size, dtype=np.float32)
        num_ellipses = np.random.randint(4, 7)
        num_noise_ellipses = np.random.randint(15, 20)
        y = -1

        if i % 2:
            y = 1
            for j in range(num_ellipses):
                ell_img = np.zeros_like(img)
                rr, cc = ellipse(np.random.randint(25, 220),
                                 np.random.randint(25, 220),
                                 np.random.randint(4, 15),
                                 np.random.randint(4, 15))

                ell_img[rr, cc] = np.random.randint(40, 80) / 100.0
                img += rotate(ell_img, np.random.randint(0, 90), clip=True, preserve_range=True)

        for j in range(num_noise_ellipses):
            ell_img = np.zeros_like(img)
            rr, cc = ellipse(np.random.randint(25, 220),
                             np.random.randint(25, 220),
                             np.random.randint(1, 8),
                             np.random.randint(1, 8))

            ell_img[rr, cc] = np.random.randint(40, 80) / 100.0
            img += rotate(ell_img, np.random.randint(0, 90), clip=True, preserve_range=True)

        n_img = np.zeros_like(img)
        img_n = random_noise(n_img, 'gaussian', None, True, var=0.15) + random_noise(n_img, 'gaussian', None, True,
                                                                                     var=0.08)
        img_n = np.array(img_n, dtype=np.float32)
        img_n = cv2.medianBlur(img_n, 5).astype(np.float64)

        binary = 1 * (img_n > (np.random.randint(5, 15) / 100.0)).astype(np.uint8)
        binary = cv2.dilate(binary, (5, 5), iterations=2)

        if i < 8:
            # w_name_str = "Synthetic image"
            # cv2.namedWindow(w_name_str, cv2.WINDOW_NORMAL)
            # cv2.resizeWindow(w_name_str, 1400, 400)
            # cv2.imshow(w_name_str, img + np.multiply(img_n, binary) )
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            out_image = img + np.multiply(img_n, binary)
            out_image /= np.max(out_image)
            cv2.imwrite("/tmp/second_stage_image_{:02d}.png".format(i), (255 * out_image).astype(np.uint8))

        images.append(np.copy(img + np.multiply(img_n, binary)))
        labels.append(y)

    return images, labels


def plot_dataset(dataset, labels, inputfiles, Tpca=None):
    fsuffix = ''
    if Tpca is None:
        Tpca = PCA(n_components=2)
        Xpca = Tpca.fit_transform(dataset)
        fsuffix = '_fit'
    else:
        Xpca = Tpca.transform(dataset)
        fsuffix = '_apply'

    zero_class = np.where(np.array(labels) < 0)
    one_class = np.where(np.array(labels) > 0)

    fig, ax = plt.subplots()

    ax.scatter(Xpca[zero_class, 0], Xpca[zero_class, 1], s=40, facecolors='b', edgecolors='b')
    ax.scatter(Xpca[one_class, 0], Xpca[one_class, 1], s=40, facecolors='r', edgecolors='r')

    for i, li in enumerate(labels):
        if li > 0:
            ax.annotate(str(i), xy=(Xpca[i, 0], Xpca[i, 1]))
            print("[Point {}] File: {}".format(i, inputfiles[i]))


    # plt.show()
    if os.environ.get('DISPLAY', '') == '':
        plt.savefig('/local/temporary/clf_vis/pca_vis{}.png'.format(fsuffix), format='png')
    else:
        plt.show()

    return Tpca


if __name__ == "__main__":

    images = []
    labels = []

    vimages = []
    vlabels = []

    if sys.argv[1] == 'syn':
        generate = True
    else:
        generate = False

    if generate:
        images, labels = generate_images()

    else:
        base_dir = "/local/temporary/clf_vis/cam16_256p_firststage"
        prefix1 = "s256patch__rep[01]__l0_vis_cam16"
        prefix2 = "s256patch__fulltrain_rep[1]__l0_vis_cam16"
        input_files_normal = sorted(glob(base_dir + "/" + prefix1 + "_normal*predictionscores_1.png"))
        input_files_tumor = sorted(glob(base_dir + "/" + prefix1 + "_tumor*predictionscores_1.png"))

        input_files_validation = sorted(glob(base_dir + "/" + prefix2 + "_annot_testing*predictionscores_2.png"))

        factor = 0.25
        im_offset = 0
        im_factor = 1 * 0.01

        for tf in input_files_tumor:
            print(" >> Read {}".format(tf))
            im = np.clip(im_factor * (cv2.imread(tf, cv2.IMREAD_GRAYSCALE) + im_offset), 0.0, 1.0)
            im2 = np.zeros((int(factor * im.shape[0]), int(im.shape[1] * factor)), dtype=np.float)
            cv2.resize(im, dst=im2, dsize=None, fx=factor, fy=factor)
            images.append(im2)
            labels.append(1)

        for nf in input_files_normal:
            print(" >> Read {}".format(nf))
            im = np.clip(im_factor * (cv2.imread(nf, cv2.IMREAD_GRAYSCALE) + im_offset), 0.0, 1.0)
            im2 = np.zeros((int(factor * im.shape[0]), int(im.shape[1] * factor)), dtype=np.float)
            cv2.resize(im, dst=im2, dsize=None, fx=factor, fy=factor) + im_offset
            images.append(im2)
            labels.append(-1)

        for vf in input_files_validation:
            print(" >> Read {}".format(vf))
            im = np.clip(im_factor * (cv2.imread(vf, cv2.IMREAD_GRAYSCALE) + im_offset), 0.0, 1.0)
            im2 = np.zeros((int(factor * im.shape[0]), int(im.shape[1] * factor)), dtype=np.float)
            cv2.resize(im, dst=im2, dsize=None, fx=factor, fy=factor) + im_offset
            vimages.append(im2)

            image_id = os.path.basename(vf).split('_')[10]
            if os.path.exists('/datagrid/Medical/microscopy/CAMELYON16/testing/Masks/test_' + image_id + '.xml'):
                vlabels.append(1)
            else:
                vlabels.append(-1)

    sigmas = [1.0, 2.0, 4.0]
    thresholds = [0.51, 0.6, 0.75, 0.8, 0.9, 0.92, 0.95]
    gr_sizes = np.arange(1, 12, 3)

    print("==[[ Classification Gaussian]]==")
    Ximg = compute_gaussian_filterings(images, sigmas, thresholds)
    Xvali = compute_gaussian_filterings(vimages, sigmas, thresholds)

    mms = MinMaxScaler()

    Ximg = mms.fit_transform(Ximg)
    Xvali = mms.transform(Xvali)

    in_pca = plot_dataset(Ximg, labels, input_files_tumor + input_files_normal)
    plot_dataset(Xvali, vlabels, input_files_validation, in_pca)

    clf_candidates = [
        RandomForestClassifier(n_jobs=6, n_estimators=30, max_depth=10),
        SVC(C=1.0, gamma=0.1, probability=True)
    ]

    Ximg = np.array(Ximg)
    labels = np.array(labels)

    for clf_max in clf_candidates:

        test_auc = []
        test_f1 = []

        skf = StratifiedKFold(n_splits=6)
        for train_idx, vali_index in skf.split(Ximg, labels):

            X_tr, X_val = Ximg[train_idx], Ximg[vali_index]
            Y_tr, Y_val = labels[train_idx], labels[vali_index]

            clf_max.fit(X_tr, Y_tr)

            val_pproba = clf_max.predict_proba(X_val)[:, 1]
            val_plabs = clf_max.predict(X_val)

            test_pproba = clf_max.predict_proba(Xvali)[:, 1]
            test_plabs = clf_max.predict(Xvali)

            test_auc.append(roc_auc_score(vlabels, test_pproba))
            test_f1.append(f1_score(vlabels, test_plabs))

        print("AUC: (test) {} +- {}".format(np.mean(test_auc), np.var(test_auc)))
        print("F1s: (test) {} +- {}".format(np.mean(test_f1), np.var(test_f1)))
        print("----------\n")


    #
    # #
    # print("==[[ Classification fGaussian]]==")
    # Ximg2 = compute_filterings_of_gaussians(images, sigmas, thresholds)
    # Xvali2 = compute_filterings_of_gaussians(vimages, sigmas, thresholds)
    # # # Classification output
    # #
    # # Xpca = PCA(n_components=2).fit_transform(Ximg2)
    # # zero_class = np.where(np.array(labels) < 0)
    # # one_class = np.where(np.array(labels) >0)
    # #
    # # plt.scatter(Xpca[zero_class, 0], Xpca[zero_class, 1], s=40, facecolors='none', edgecolors='b')
    # # plt.scatter(Xpca[one_class, 0], Xpca[one_class, 1], s=40, facecolors='none', edgecolors='r')
    # # plt.show()
    # #
    # clf_max = RandomForestClassifier(n_jobs=6, n_estimators=30, max_depth=10)
    # sc = cross_val_score(clf_max, Ximg2, labels, scoring='f1_macro', cv=5)
    # print("F1: %0.4f (+/- %0.4f)" % (sc.mean(), sc.std() * 2))
    #
    # clf_max.fit(Ximg, labels)
    # pproba = clf_max.predict_proba(Xvali)[:, 1]
    # plabs = clf_max.predict(Xvali)
    #
    # print("AUC: " + str(roc_auc_score(vlabels, pproba)))
    # print("F1s: " + str(f1_score(vlabels, plabs)))
    # # print("==[[ Classification Granulometric ]]==")
    # # Ximg_gr = compute_granulometry_features(images, sigmas, thresholds, gr_sizes)
    # # sc = cross_val_score(clf_max, Ximg_gr, labels, scoring='f1_macro', cv=5)
    # # print("F1: %0.4f (+/- %0.4f)" % (sc.mean(), sc.std() * 2))
