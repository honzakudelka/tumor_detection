import cv2
import numpy as np
from skimage import feature
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


def haralick_features(img_gray):
    n_clusters = 16
    im = np.array(img_gray, dtype=np.float64) / 255
    arr = im.reshape((img_gray.shape[0] * img_gray.shape[1], img_gray.shape[2]))

    arr_sample = shuffle(arr, random_state=0)[:50000]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(arr_sample)
    labels = kmeans.predict(arr)
    labels = labels.reshape((img_gray.shape[0], img_gray.shape[1]))

    D = feature.greycomatrix(labels, [1, 2, 5, 10, 20], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], n_clusters)
    contrast = feature.greycoprops(D, 'contrast')
    homogeneity = feature.greycoprops(D, 'homogeneity')
    energy = feature.greycoprops(D, 'energy')
    dissimilarity = feature.greycoprops(D, 'dissimilarity')

    Df = (np.clip(D, [0., 0., 0., 0.], [1020., 1020., 1020., 1020.]) / 4).astype(np.uint8)
    D_feat = [homogeneity[0], contrast[0], energy[0], dissimilarity[0]]
    D_feat_names = ['greyco_hom', 'greyco_con', 'greyco_en', 'greyco_dis']

    H = {}
    for i in range(0, 4):

        w_str = "Haralick features at distances [1, 5, 10]"
        cv2.namedWindow(w_str, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(w_str, 900, 300)
        cv2.imshow(w_str, np.hstack([Df[:, :, 0, i], Df[:, :, 1, i], Df[:, :, 2, i]]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for j in range(0, 4):
            H.update({"{0}_{1:d}".format(D_feat_names[i], j): D_feat[i][j]})

    return H


def color_channel_histogram(img):
    hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])

    pixels = 1.0 * img.shape[0] * img.shape[1]

    features_dict = {}
    for i in range(len(hist_b)):
        features_dict.update({"HIST_R{0:02d}".format(i): hist_r[i][0] / pixels})
        features_dict.update({"HIST_G{0:02d}".format(i): hist_g[i][0] / pixels})
        features_dict.update({"HIST_B{0:02d}".format(i): hist_b[i][0] / pixels})

    return features_dict


def _filterh_y(image, l):
    imff = 0.5 * np.pad(image, ((0, l), (0, 0)), 'reflect')
    imff = imff[:-l, :] + imff[l:, :]

    return imff


def _filterg_y(image, l):
    imff = 0.5 * np.pad(image, ((0, l), (0, 0)), 'reflect')
    imff = imff[l:, :] - imff[:-l, :]

    return imff


def _filterh_x(image, l):
    imff = 0.5 * np.pad(image, ((0, 0), (0, l)), 'reflect')
    imff = imff[:, :-l] + imff[:, l:]

    return imff


def _filterg_x(image, l):
    imff = 0.5 * np.pad(image, ((0, 0), (0, l)), 'reflect')
    imff = imff[:, l:] - imff[:, :-l]

    return imff


def get_wavelet_responses(image, prefix='', subdivision=(1, 1), downsample=1, maxlevel=5, bin_min=1e-4, bin_max=5e-2,
                          separate=False):
    source_image = image
    if downsample < 1:

        source_image = cv2.GaussianBlur(source_image, ksize=(0, 0),
                                        sigmaX=1.0 / downsample, sigmaY=1.0 / downsample)

        subpatch_size_x = source_image.shape[0]
        subpatch_size_y = source_image.shape[1]
        n_patches = 1

    else:
        subpatch_size_x = image.shape[0] // subdivision[0]
        subpatch_size_y = image.shape[1] // subdivision[1]

        n_patches = subdivision[0] * subdivision[1]

    vfac = 1e1
    responses = np.zeros((n_patches, maxlevel + 1))
    lresponses = []
    for x in range(subdivision[0]):
        for y in range(subdivision[1]):

            timage = source_image[x * subpatch_size_x:(x + 1) * subpatch_size_x,
                     y * subpatch_size_y:(y + 1) * subpatch_size_y]

            for i in range(maxlevel):
                l = 2 ** (i + 1)
                imhy = _filterh_y(timage, l)
                imgy = _filterg_y(timage, l)

                vgg_im = _filterg_x(imgy, l)
                vhg_im = _filterh_x(imgy, l)
                vgh_im = _filterg_x(imhy, l)

                timage = np.transpose(np.copy(_filterh_x(imhy, l)))

                vgg_pow = vfac * np.mean(np.power(vgg_im, 2))
                vhg_pow = vfac * np.mean(np.power(vhg_im, 2))
                vgh_pow = vfac * np.mean(np.power(vgh_im, 2))
                responses[x * subdivision[1] + y, i] = vgg_pow + vhg_pow + vgh_pow
                if separate:
                    lresponses.append([vgg_pow + vhg_pow + vgh_pow, vgg_pow, vhg_pow, vgh_pow])
                else:
                    lresponses.append([vgg_pow + vhg_pow + vgh_pow])

                # w_name_str = "|{:.3f}|{:.3f}|{:.3f}|".format(vgg_pow, vhg_pow, vgh_pow)
                # cv2.namedWindow(w_name_str, cv2.WINDOW_NORMAL)
                # cv2.resizeWindow(w_name_str, 1400, 400)
                # cv2.imshow(w_name_str, np.hstack([ source_image,
                #                                    timage,
                #                                    vgg_im,
                #                                    vhg_im,
                #                                    vgh_im
                #                                   ]
                #                                 )
                #            )
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            # lp-energy
            responses[x * subdivision[1] + y, maxlevel] = np.mean(np.power(timage, 2))
            lresponses.append([np.mean(np.power(timage, 2))])

    n_bins = 16
    w_hist = np.zeros(n_bins)
    w_lphist = np.zeros(n_bins)
    bin_space = np.logspace(start=np.log10(bin_min), stop=np.log10(bin_max), num=n_bins + 1)
    lp_bin_space = np.logspace(start=np.log10(1e-3), stop=np.log10(2), num=n_bins + 1)
    for l in range(maxlevel):
        l_hist, l_bins = np.histogram(responses[:, l], bins=bin_space)
        w_hist += l_hist * (1. / n_patches)

    lphist, lpbins = np.histogram(responses[:, maxlevel], bins=lp_bin_space)
    w_lphist += lphist * (1. / n_patches)

    wx_features = dict()
    if n_patches > 1:
        for bi, hbin in enumerate(w_hist):
            wx_features.update({"wx_his_" + prefix + "bin_{0:02d}".format(bi): hbin})

        for bi, hbin in enumerate(w_lphist):
            wx_features.update({"wx_hls_" + prefix + "bin_{0:02d}".format(bi): hbin})
    else:
        for ri, response in enumerate(lresponses):
            for rii, resp_val in enumerate(response):
                wx_features.update({"wx_res_" + prefix + "_ft_{0:02d}_lev_{1:02d}".format(rii, ri): resp_val})

    return wx_features


def get_gabor_filterbank_responses(image):
    fimage = np.float32(image)

    theta_val = [0, 30, 60, 90]
    freq = [2, 4, 6, 8, 12, 16, 24, 32]

    responses = np.zeros((len(freq), 3))
    for k, lam in enumerate(freq):
        for theta in theta_val:
            sig = max(2.0, lam * 0.5)

            s_gab = cv2.getGaborKernel((-1, -1), sig, theta * np.pi / 180.0, lam, 0.75, 0.5 * np.pi, cv2.CV_32F)
            c_gab = cv2.getGaborKernel((-1, -1), sig, theta * np.pi / 180.0, lam, 0.75, 0.0, cv2.CV_32F)

            s_response = cv2.filter2D(fimage, -1, s_gab)
            c_response = cv2.filter2D(fimage, -1, c_gab)
            mag_res = cv2.magnitude(c_response[:, :], s_response[:, :])

            # plt.subplot(221), plt.imshow(fimage, cmap='gray')
            # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
            # # plt.subplot(232), plt.imshow(s_response, cmap='gray')
            # # plt.title('SIN'), plt.xticks([]), plt.yticks([])
            # # plt.subplot(233), plt.imshow(c_response, cmap='gray')
            # # plt.title('COS'), plt.xticks([]), plt.yticks([])
            # #
            # plt.subplot(223), plt.imshow(s_gab, cmap='gray')
            # plt.title('SIN'), plt.xticks([]), plt.yticks([])
            # plt.subplot(224), plt.imshow(c_gab, cmap='gray')
            # plt.title('COS'), plt.xticks([]), plt.yticks([])
            # #
            # #
            # plt.subplot(222), plt.imshow(mag_res, cmap='gray')
            # plt.title('MAG'), plt.xticks([]), plt.yticks([])
            # plt.suptitle("Theta = {0:.1f} | Freq = {1:.1f}".format(theta, lam))
            # plt.show()
            #

            # print("Processing: s = {}, lam = {}, theta = {}".format(sig, lam, theta))
            # print("GABOR({},{},{}): [{}] {} +- {}  [{} {}]".format(sig, lam, theta,
            #                                                        mag_res.sum(),
            #                                                        mag_res.mean(), mag_res.var(),
            #                                                        mag_res.min(), mag_res.max()))

            mag_res = np.array(mag_res)
            responses[k] += [mag_res.mean(), mag_res.var(), mag_res.min() / mag_res.max()]

    tx_features = {}
    # print("[ID] : MEAN   VAR    MMR")
    for ri, resp in enumerate(responses):
        # print("[{}] : {} {} {}".format(ri, resp[0], resp[1], resp[2]))
        tx_features.update({"tx_kerfreq_{0:d}_mean".format(ri): resp[0],
                            "tx_kerfreq_{0:d}_var".format(ri): resp[1],
                            "tx_kerfreq_{0:d}_mmr".format(ri): resp[2]})

    return tx_features


def compute_textural_features(image):
    # prepare filter bank kernels
    fimage = image.copy()
    fimage = np.float32(fimage)
    fimage = np.float32(1.0 * (fimage - np.mean(fimage)) / np.var(fimage))
    fimage = fimage / (np.max(fimage) - np.min(fimage))

    tx_dict = dict()
    # tx_dict = get_wavelet_responses(fimage)
    tx_dict.update(get_gabor_filterbank_responses(fimage))

    return tx_dict
