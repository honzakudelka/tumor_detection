from scipy import ndimage

import cv2
import numpy as np
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label
from skimage.morphology import watershed
from sklearn.cluster import MiniBatchKMeans


class StructuralFeatures:
    def __init__(self):
        self.fullpatch_x = 1024
        self.patch_x = 256
        self.histogram = None
        self.num_nuclei = -1
        self.subpatch_histogram = []
        self.lut = [0, 1, 2, 3]

    @property
    def get_global_features(self):

        if self.histogram is None:
            raise RuntimeError("No histogram set.")

        area_inv = 1.0 / (self.fullpatch_x * self.fullpatch_x)

        i0 = self.lut[0]
        i1 = self.lut[1]
        i2 = self.lut[2]
        i3 = self.lut[3]

        global_features = {'num_nuclei': self.num_nuclei,
                           'rel_area_stroma': self.histogram[i2] * area_inv,
                           'rel_area_lumen': self.histogram[i1] * area_inv,
                           'rel_area_nuclei': self.histogram[i0] * area_inv,
                           'rel_area_background': self.histogram[i3] * area_inv,
                           'rel_ratio_stroma_nucl': 1. * self.histogram[i2] / self.histogram[i0],
                           'rel_ratio_stroma_lumen': 1. * self.histogram[i2] / self.histogram[i1],
                           'rel_ratio_nuclei_lumen': 1. * self.histogram[i0] / self.histogram[i1]}

        return global_features

    @property
    def get_subpatch_features(self):

        if len(self.subpatch_histogram) < 1:
            raise RuntimeError("No subpatch histogram set.")

        S_tot = 1.0 * self.patch_x ** 2

        self.subpatch_histogram = np.array(self.subpatch_histogram)
        area_inv = 1.0 / (self.fullpatch_x * self.fullpatch_x)

        i0 = self.lut[0]
        i1 = self.lut[1]
        i2 = self.lut[2]
        i3 = self.lut[3]

        var_nuclei = np.var(self.subpatch_histogram[:, i0] / S_tot)
        mean_nuclei = np.mean(self.subpatch_histogram[:, i0] / S_tot)

        var_lumen = np.var(self.subpatch_histogram[:, i1] / S_tot)
        mean_lumen = np.mean(self.subpatch_histogram[:, i1] / S_tot)

        var_stroma = np.var(self.subpatch_histogram[:, i2] / S_tot)
        mean_stroma = np.mean(self.subpatch_histogram[:, i2] / S_tot)

        var_nuc_lum = np.var(1.0 * self.subpatch_histogram[:, i0] / self.subpatch_histogram[:, i1])
        mean_nuc_lum = np.mean(1.0 * self.subpatch_histogram[:, i0] / self.subpatch_histogram[:, i1])

        var_str_lum = np.var(1.0 * self.subpatch_histogram[:, i2] / self.subpatch_histogram[:, i1])
        mean_str_lum = np.mean(1.0 * self.subpatch_histogram[:, i2] / self.subpatch_histogram[:, i1])

        var_str_nuc = np.var(1.0 * self.subpatch_histogram[:, i2] / self.subpatch_histogram[:, i0])
        mean_str_nuc = np.mean(1.0 * self.subpatch_histogram[:, i2] / self.subpatch_histogram[:, i0])

        subpatch_features = {
            'sp_var_nuclei': var_nuclei,
            'sp_mean_nuclei': mean_nuclei,
            'sp_var_lumen': var_lumen,
            'sp_mean_lumen': mean_lumen,
            'sp_var_stroma': var_stroma,
            'sp_mean_stroma': mean_stroma,
            'sp_var_nuc_lum': var_nuc_lum,
            'sp_mean_nuc_lum': mean_nuc_lum,
            'sp_var_str_lum': var_str_lum,
            'sp_mean_str_lum': mean_str_lum,
            'sp_var_str_nuc': var_str_nuc,
            'sp_mean_str_nuc': mean_str_nuc,
            'sprel_area_nuclei': np.sum(self.subpatch_histogram[:, i0]) * area_inv,
            'sprel_area_lumen': np.sum(self.subpatch_histogram[:, i1]) * area_inv,
            'sprel_area_stroma': np.sum(self.subpatch_histogram[:, i2]) * area_inv,
        }

        return subpatch_features


def he_structural_features(imH, imE, imR, prefix='', subdivision=(1, 1)):
    # patch_size = imH.shape[:2]
    # subpatch_x = patch_size[0]//subpatch[0]
    # subpatch_y = patch_size[1]//subpatch[1]

    background = (1 * (imH + imE + (255 - imR) < 40)).astype(np.uint8)
    tissue_E = (1 * (imE > 40)).astype(np.uint8)
    tissue_H = (1 * (imH > 60)).astype(np.uint8)
    # tissue_mixed = (255 * (imE > 40) & ( imH < 60)).astype(np.uint8)

    # visE, keypoints = blob_detection_watershed(np.copy(background), 1, np.copy(patch),
    #                                            area_low=200, area_high=1000,
    #                                            skip_watershed=True)
    #
    # w_name_str = "Structural features from H&E"
    # cv2.namedWindow(w_name_str, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(w_name_str, 1400, 400)
    # cv2.imshow(w_name_str, np.hstack([ cv2.cvtColor(255*background,cv2.COLOR_GRAY2RGB),
    #                                    cv2.cvtColor(255*tissue_E, cv2.COLOR_GRAY2RGB),
    #                                    cv2.cvtColor(255*tissue_H, cv2.COLOR_GRAY2RGB)
    #                                   ]
    #                                 )
    #            )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    subpatch_size_x = imH.shape[0] // subdivision[0]
    subpatch_size_y = imH.shape[1] // subdivision[1]

    n_patches = subdivision[0] * subdivision[1]

    features = {
        prefix + "he_stru_bck_area": np.mean(background),
        prefix + "he_stru_hem_area": np.mean(tissue_H),
        prefix + "he_stru_eos_area": np.mean(tissue_E),
    }

    if n_patches > 1:

        responses = np.zeros((n_patches, 3))
        for x in range(subdivision[0]):
            for y in range(subdivision[1]):
                bimage = background[x * subpatch_size_x:(x + 1) * subpatch_size_x,
                         y * subpatch_size_y:(y + 1) * subpatch_size_y]

                himage = tissue_E[x * subpatch_size_x:(x + 1) * subpatch_size_x,
                         y * subpatch_size_y:(y + 1) * subpatch_size_y]

                eimage = tissue_H[x * subpatch_size_x:(x + 1) * subpatch_size_x,
                         y * subpatch_size_y:(y + 1) * subpatch_size_y]

                responses[x * subdivision[1] + y, :] = [np.mean(bimage), np.mean(himage), np.mean(eimage)]

        features.update({
            prefix + "he_stru_bck_var": np.var(responses[:, 0]),
            prefix + "he_stru_hem_var": np.var(responses[:, 1]),
            prefix + "he_stru_eos_var": np.var(responses[:, 2])
        })

    return features

    # for dx in range(subpatch_x):
    #     sx = dx * subpatch[0]
    #     for dy in range(subpatch_y):
    #         sy = dy * subpatch[1]
    #
    #         # trust the quantization on whole-slide level, take the labels as are
    #         #   (and save processing time)
    #         #
    #         subpatch_data = image[sx:sx+subpatch[0], sy:sy+subpatch[1], :]


def blob_detection_watershed(image, scale=1, vis_image=None, area_low=100, area_high=700, skip_watershed=False):
    image_8bit = image.copy()
    if len(image.shape) > 2 and image.shape[2] > 1:
        image_8bit = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if not skip_watershed:
        D = ndimage.distance_transform_edt(image_8bit)
        localMax = peak_local_max(D, indices=False, min_distance=8, labels=image_8bit)
        markers = ndimage.label(localMax)[0]
        wlabels = watershed(-D, markers, mask=image_8bit)
    else:
        wlabels = label(image_8bit) + 1

    wmin, wmax, _, _ = cv2.minMaxLoc(wlabels)

    regions = regionprops(wlabels)

    im_with_keypoints = None
    if vis_image is not None:
        # im_with_keypoints = vis_image
        im_with_keypoints = label2rgb(wlabels, image=vis_image)

    keypoints = []
    for region in regions:

        x0, y0 = region.centroid
        r = int(0.5 * region.equivalent_diameter)
        ecc = region.eccentricity

        a = region.filled_area

        if (area_low < a * (scale ** 2) < area_high) and (region.solidity > 0.8 or ecc < 0.7):
            if vis_image is not None:
                cv2.circle(im_with_keypoints, (int(y0), int(x0)), r, (125, 255, 0), 2)

            keypoints.append((x0, y0))
            # else:
            #     if vis_image is not None:
            #         # if area_low < a * (scale**2) < area_high:
            #         #     cv2.circle(im_with_keypoints, (int(y0), int(x0)), r, (0, 255, 0))
            #         # else:
            #         #     cv2.circle(im_with_keypoints, (int(y0), int(x0)), r, (255, 0, 255))

    return im_with_keypoints, keypoints


def blob_detection_lindeberg(image, vis_image=None):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 256

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    # params.maxArea = 500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1

    # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(image)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = image
    if vis_image is not None:
        im_with_keypoints = vis_image

    im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, keypoints, np.array([]), (125, 255, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return im_with_keypoints, keypoints


def color_quantization_cv(img, nbCluster):
    # Color quantization with OpenCVPython
    h, w = img.shape[:2]
    Z = img.reshape((h * w, -1))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = nbCluster  # background, stroma, lumina and nuclei
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    idx_tr = np.array([x for _, x in sorted(zip(np.sum(center, axis=1), [0, 1, 2, 3]))], dtype="uint8")

    index_lut = np.zeros(4)
    for i, j in enumerate(idx_tr):
        index_lut[j] = i

    label_image = index_lut[label.flatten()]

    image_nc = np.copy(label_image)
    label_image = label_image.reshape((w, h))

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape(img.shape)
    # quant = cv2.cvtColor(res, cv2.COLOR_RGBA2BGR)

    image_nc[image_nc > 0] = 255
    image_nc = 255 - image_nc
    image_nc = image_nc.reshape((w, h))

    return res, image_nc, label_image


def color_quantization(image, n_cluster):
    """
    K-means clustering of the input image

    :param image: input image
    :param n_cluster: number of required clusters
    :return: clustered image
    """
    h, w = image.shape[:2]

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    Z = lab_image.reshape((h * w, lab_image.shape[2]))
    # Z = np.float32(Z)

    clt = MiniBatchKMeans(n_clusters=n_cluster, batch_size=1024)
    labels = clt.fit_predict(Z)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # unified coloring
    colors = np.array([[255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 0, 0]])
    blob_colors = np.array([[255, 255, 255] for _ in range(n_cluster)])
    blob_colors[0] = [0, 0, 0]

    idx_tr = np.array([x for _, x in sorted(zip(np.sum(clt.cluster_centers_, axis=1), [0, 1, 2, 3]))], dtype="uint8")

    index_lut = np.zeros(4)
    for i, j in enumerate(idx_tr):
        index_lut[j] = i

    label_image = index_lut[labels]

    quant = quant.reshape((h, w, 3))
    image_nc = np.copy(label_image)
    image_nc[image_nc > 0] = 255
    image_nc = np.uint8(255 - image_nc)

    image_nc = image_nc.reshape((w, h))
    label_image = label_image.reshape((h, w))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

    return quant, image_nc, label_image


def get_structural_features(image, subpatch, level=0, visualize=False, detect_nuclei=True):
    n_clusters = 4

    img_Q, img_Nc, labels = color_quantization(image, n_clusters)

    img_w_vis = None

    if visualize:
        img_w_vis = cv2.cvtColor(image.copy(), cv2.COLOR_RGBA2BGR)

    if detect_nuclei:
        img_Nc_w, img_Nc_w_keypoints = blob_detection_watershed(img_Nc, pow(2, level), img_w_vis)

    if visualize:
        cv2.namedWindow("Nuclei detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Nuclei detection", 1400, 900)
        if detect_nuclei:
            cv2.imshow("Nuclei detection", np.hstack([img_Q,
                                                      cv2.cvtColor(img_Nc, cv2.COLOR_GRAY2RGB),
                                                      img_Nc_w]))
        else:
            cv2.imshow("Nuclei detection", np.hstack([img_Q,
                                                      cv2.cvtColor(img_Nc, cv2.COLOR_GRAY2RGB),
                                                      image]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # histogram
    sf = StructuralFeatures()
    sf.histogram, _ = np.histogram(labels, bins=[0, 1, 2, 3, 4])
    sf.num_nuclei = -1
    if detect_nuclei:
        sf.num_nuclei = len(img_Nc_w_keypoints)

    if np.sum(sf.histogram[1:]) < 1:
        return None

    global_features = sf.get_global_features

    patch_size = image.shape[:2]
    subpatch_x = patch_size[0] / subpatch[0]
    subpatch_y = patch_size[1] / subpatch[1]

    for dx in range(subpatch_x):
        sx = dx * subpatch[0]
        for dy in range(subpatch_y):
            sy = dy * subpatch[1]

            # trust the quantization on whole-slide level, take the labels as are
            #   (and save processing time)
            #
            subpatch_data = image[sx:sx + subpatch[0], sy:sy + subpatch[1], :]
            sp_labels = labels[sx:sx + subpatch[0], sy:sy + subpatch[1]]

            bck = _automask_background(subpatch_data)
            pixels = subpatch_data.shape[0] * subpatch_data.shape[1]
            coverage = sum(bck.reshape((1, pixels))[0]) / 255.0
            if 0.95 * pixels < coverage:
                n_clusters = 3

                sp_Q, sp_Nc, sp_labels_refit = color_quantization(subpatch_data, n_clusters)
                sp_lab_combined = np.minimum(sp_labels, sp_labels_refit)
            else:
                sp_lab_combined = sp_labels

            s_hist, s_bins = np.histogram(sp_lab_combined, bins=[0, 1, 2, 3, 4])
            sf.subpatch_histogram.append(s_hist)

    subpatch_structural = sf.get_subpatch_features

    return dict(global_features, **subpatch_structural)


def _automask_background(image, boundaries=([0, 0, 0], [220, 220, 220])):
    # loop over the boundaries

    # create NumPy arrays from the boundaries
    lower = np.array(boundaries[0], dtype="uint8")
    upper = np.array(boundaries[1], dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    imgray = cv2.GaussianBlur(cv2.cvtColor(output, cv2.COLOR_BGR2GRAY), (3, 3), 2)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY)

    return cv2.erode(thresh, (3, 3))
