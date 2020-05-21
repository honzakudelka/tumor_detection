import random

import numpy as np
from tflearn import ImageAugmentation, DataPreprocessing

from hp_utils import background_subtraction

_EPSILON = 1e-8

rgb_from_her = np.array([[0.4605, 0.7538, 0.3914],
                         [0.2948, 0.7491, 0.5873],
                         [0.2720, 0.8782, 0.3852]])
her_from_rgb = None

in_modulation_matrix = np.array([[0.02, 0.1, 0.1],
                                 [0.05, 0.02, 0.05],
                                 [0.05, 0.05, 0.02]])


class HEAugmentation(ImageAugmentation):
    def __init__(self):
        super(HEAugmentation, self).__init__()

    def add_random_he_variation(self, alpha, mean, stdev):
        global her_from_rgb, rgb_from_her

        if her_from_rgb is None:
            her_from_rgb = np.linalg.inv(rgb_from_her)

        self.methods.append(self._he_variation)
        self.args.append([alpha, mean, stdev])

    @staticmethod
    def _he_variation(batch, alpha, c_mean, c_std):

        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                restored_img = batch[i] * (c_std + _EPSILON) + c_mean
                batch_i_hed = _he_separation_skimage(restored_img, alpha)
                scaled_augmented = (batch_i_hed - c_mean) / (c_std - _EPSILON)

                batch[i] = scaled_augmented

        return batch


def _get_he_image_randsearch(im):
    global her_from_rgb, rgb_from_her, in_modulation_matrix

    n_run = 0
    min_runs = 3
    max_runs = 15

    min_res = 10
    best_mixing = rgb_from_her
    best_mixing_inv = None

    rgb = 1 / 255.0 * (np.copy(im).astype(np.float32) + 1)
    rgb = rgb.reshape((-1, 3))
    rgb[:, [0, 2]] = rgb[:, [2, 0]]

    limit = 1e-7

    modulation_matrix = np.copy(in_modulation_matrix)
    while True:

        a_rgb_from_her = rgb_from_her + np.multiply(modulation_matrix, (3 * np.random.rand(*rgb_from_her.shape) - 1.5))
        a_her_from_rgb = np.linalg.inv(a_rgb_from_her)

        stains = np.dot(-np.log(rgb), a_her_from_rgb)
        r_stains = np.zeros_like(stains)
        r_stains[:, 0] = stains[:, 0]

        im_R = 1 - np.exp(np.dot(-r_stains, a_rgb_from_her))
        im_R = np.clip(im_R, 0, 1)
        score_var = np.var(im_R)
        score_mean = np.mean(im_R)

        score = np.abs(score_mean) * score_var * 10
        if n_run < max_runs and (n_run < min_runs or not (score < limit)):
            improvement = False

            if min_res > score:
                min_res = score
                best_mixing_inv = a_her_from_rgb
                best_mixing = a_rgb_from_her
                id_best = n_run
                improvement = True

            if improvement and score < limit * 1e2:
                rgb_from_her = np.copy(a_rgb_from_her)
                modulation_matrix = 0.9 * modulation_matrix

            if n_run > max_runs / 2 and min_res < limit * 1e2:
                try_variations = False
                break

            n_run += 1
            continue

        else:

            if min_res > score:
                min_res = score
                best_mixing_inv = a_her_from_rgb
                best_mixing = a_rgb_from_her
                id_best = n_run

            break

    im_out = np.zeros_like(rgb)
    stains = np.dot(-np.log(rgb), best_mixing_inv)

    r_stains = np.zeros_like(stains)
    h_stains = np.zeros_like(stains)
    e_stains = np.zeros_like(stains)

    r_stains[:, 0] = np.copy(stains[:, 0])
    im_out[:, 0] = np.mean(np.clip(np.exp(np.dot(-r_stains, best_mixing)),
                                   0, 1), axis=1)

    h_stains[:, 1] = np.copy(stains[:, 1])
    im_out[:, 1] = np.mean(np.clip((1 - np.exp(np.dot(-h_stains, best_mixing))),
                                   0, 1), axis=1)

    e_stains[:, 2] = np.copy(stains[:, 2])
    im_out[:, 2] = np.mean(np.clip((1 - np.exp(np.dot(-e_stains, best_mixing))),
                                   0, 1), axis=1)

    # im_out.reshape( im.shape )
    return im_out


def _he_separation_skimage(im, alpha):
    global her_from_rgb, rgb_from_her

    rgb = 1 / 255.0 * (np.copy(im).astype(np.float32) + 1)
    rgb = rgb.reshape((-1, 3))
    rgb[:, [0, 2]] = rgb[:, [2, 0]]

    stains = np.dot(-np.log(rgb), her_from_rgb)

    augm_alpha = 1.0 + alpha * np.random.randint(-10, 10, 3)
    augm_beta = alpha * np.random.randint(-10, 10, 3)

    augm_stains = augm_alpha * stains + augm_beta
    im_A = 255 * np.exp(np.dot(-augm_stains, rgb_from_her))

    im_A[:, [0, 2]] = im_A[:, [2, 0]]
    imA = np.reshape(np.clip(im_A, 0, 255), im.shape).astype(np.float32)

    return imA


def _get_he_image(im):
    global her_from_rgb, rgb_from_her

    rgb = 1 / 255.0 * (np.copy(im).astype(np.float32) + 1)

    rgb = rgb.reshape((-1, 3))
    im_out = np.zeros_like(rgb)
    rgb[:, [0, 2]] = rgb[:, [2, 0]]

    stains = np.dot(-np.log(rgb), her_from_rgb)

    r_stains = np.zeros_like(stains)
    h_stains = np.zeros_like(stains)
    e_stains = np.zeros_like(stains)

    r_stains[:, 0] = np.copy(stains[:, 0])
    im_out[:, 0] = np.mean(np.clip(np.exp(np.dot(-r_stains, rgb_from_her)),
                                   0, 1), axis=1) - 0.5

    h_stains[:, 1] = np.copy(stains[:, 1])
    im_out[:, 1] = np.mean(np.clip((1 - np.exp(np.dot(-h_stains, rgb_from_her))),
                                   0, 1), axis=1) - 0.5

    e_stains[:, 2] = np.copy(stains[:, 2])
    im_out[:, 2] = np.mean(np.clip((1 - np.exp(np.dot(-e_stains, rgb_from_her))),
                                   0, 1), axis=1) - 0.5

    # im_out.reshape( im.shape )
    return im_out


class HEPreprocessing(DataPreprocessing):
    def __init__(self):
        super(HEPreprocessing, self).__init__()

    def add_random_crop(self, shape):
        self.methods.append(self._crop_random)
        self.args.append([shape])

    def add_odens_transform(self, scale):
        self.methods.append(self._convert_log)
        self.args.append([scale])

    def convert_to_he_space(self):
        global her_from_rgb, rgb_from_her

        if her_from_rgb is None:
            her_from_rgb = np.linalg.inv(rgb_from_her)

        self.methods.append(self._convert_to_he)
        self.args.append(None)

    def convert_to_he_space_randsep(self):
        global her_from_rgb, rgb_from_her

        if her_from_rgb is None:
            her_from_rgb = np.linalg.inv(rgb_from_her)

        self.methods.append(self._convert_to_he_rand)
        self.args.append(None)

    def add_background_subtraction(self):
        self.methods.append(self._subtract_background)
        self.args.append(None)

    @staticmethod
    def _crop_random(batch, shape):
        oshape = np.shape(batch[0])
        nh = random.randint(0, int((oshape[0] - shape[0]) * 0.5))
        nw = random.randint(0, int((oshape[1] - shape[1]) * 0.5))
        new_batch = []
        for i in range(len(batch)):
            new_batch.append(batch[i][nh: nh + shape[0], nw: nw + shape[1]])
        return new_batch

    @staticmethod
    def _subtract_background(batch):
        new_batch = []
        for i in range(len(batch)):
            bg_sub_patch = background_subtraction.subtract_background(batch[i], light_background=True,
                                                                      radius=20, down_factor=1)
            new_batch.append(bg_sub_patch)

        return new_batch

    @staticmethod
    def _convert_log(batch, scale):
        oshape = np.shape(batch[0])
        new_batch = []
        for i in range(len(batch)):
            new_batch.append(-np.log((1 + batch[i]) / scale))

        return new_batch

    @staticmethod
    def _convert_to_he(batch):
        oshape = np.shape(batch[0])
        # print("IN Shape " + str(oshape))
        new_batch = []
        for i in range(len(batch)):
            he_im = _get_he_image(batch[i])
            # print("HE Shape " + str(he_im.shape))
            new_batch.append(he_im.reshape(oshape))

        return new_batch

    @staticmethod
    def _convert_to_he_rand(batch):
        oshape = np.shape(batch[0])
        # print("IN Shape " + str(oshape))
        new_batch = []
        for i in range(len(batch)):
            he_im = _get_he_image_randsearch(batch[i])
            # print("HE Shape " + str(he_im.shape))
            new_batch.append(he_im.reshape(oshape))

        return new_batch
