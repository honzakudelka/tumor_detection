# used in __main_ only_
# from slide_paraboloid import sliding_paraboloid_float_background
# import cp_slide_paraboloid as cp_par

import ctypes

import cv2
import numpy as np
from numpy.ctypeslib import ndpointer


def _shrink_image(im, factor):
    w = im.shape[1]
    h = im.shape[0]
    pixels = im[:]

    sw = (w + factor - 1) // factor
    sh = (h + factor - 1) // factor

    s_im = np.zeros((sh, sw))
    for y in range(sh):
        for x in range(sw):
            # downscale by reducing each neighborhood to its minimal value
            s_im[y, x] = np.amin(pixels[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor])

    return s_im


def _get_interpolation_arrays(small_indices, weights, full_len, small_len, factor):
    for i in range(full_len):
        s_ind = (i - factor // 2) // factor
        s_ind = min(s_ind, small_len - 2)
        small_indices[i] = s_ind

        s_dist = (i + 0.5) / factor - (s_ind + 0.5)
        weights[i] = 1. - s_dist


def _enlarge_image(im, factor):
    sw = im.shape[1]
    sh = im.shape[0]
    pixels = np.copy(im).flatten()

    w = sw * factor
    h = sh * factor

    l_im = np.zeros((w, h))

    x_sm_indices = np.zeros(w, dtype=np.int)
    y_sm_indices = np.zeros(h, dtype=np.int)

    x_sm_weights = np.zeros(w, dtype=np.float)
    y_sm_weights = np.zeros(h, dtype=np.float)

    _get_interpolation_arrays(x_sm_indices, x_sm_weights, w, sw, factor)
    _get_interpolation_arrays(y_sm_indices, y_sm_weights, h, sh, factor)

    line1 = np.zeros(w)
    line0 = np.zeros_like(line1)
    # fill first line
    for x in range(w):
        line1[x] = pixels[x_sm_indices[x]] * x_sm_weights[x] + \
                   pixels[x_sm_indices[x] + 1] * (1.0 - x_sm_weights[x])

    y_line0 = -1
    for y in range(h):
        if y_line0 < y_sm_indices[y]:
            line0[:], line1[:] = line1[:], line0[:]
            y_line0 += 1

            sy_position = (y_sm_indices[y] + 1) * sw
            for x in range(w):
                line1[x] = pixels[sy_position + x_sm_indices[x]] * x_sm_weights[x] + \
                           pixels[sy_position + x_sm_indices[x] + 1] * (1. - x_sm_weights[x])

        weight = y_sm_weights[y]
        for x in range(w):
            l_im[y, x] = line0[x] * weight + line1[x] * (1. - weight)

    return l_im


def subtract_background(im, light_background, radius, down_factor):
    factor = 1
    if light_background:
        factor = -1

    im_hsv = cv2.cvtColor(im[:, :, :3], cv2.COLOR_RGB2HSV)
    v_im = factor * im_hsv[:, :, 2]

    if down_factor > 1:
        im_dwn = _shrink_image(v_im, res_factor)
    else:
        im_dwn = v_im

    im_orig = np.copy(factor * v_im)
    pixels = im_dwn.flatten()
    w = im_dwn.shape[1]
    h = im_dwn.shape[0]

    lib = ctypes.cdll.LoadLibrary('cpp_slide_paraboloid.so')
    fun = lib.sliding_paraboloid_float_background
    fun.restype = None
    fpixels = pixels.astype(np.float32)
    fun.argtypes = [ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool, ctypes.c_bool]

    fun(fpixels, w, h, radius, False, False)
    pixels = fpixels.astype(pixels.dtype)

    im_dwn[:] = np.array(pixels).reshape((h, w))[:]

    if down_factor > 1:
        im_up = (factor * _enlarge_image(im_dwn, res_factor)).astype(np.uint8)
    else:
        im_up = (factor * im_dwn).astype(np.uint8)

    im_res = im_orig - im_up + 255.5
    im_res = np.clip(im_res, 0, 255).astype(np.uint8)

    res_hsv = im_hsv[:]
    res_hsv[:, :, 2] = im_res[:]

    res_im = cv2.cvtColor(res_hsv, cv2.COLOR_HSV2RGB)

    return res_im


if __name__ == "__main__":

    fname_in = "/tmp/patch_petacc.tif"
    fname_out = "/tmp/patch_petacc_processed.tif"

    im = cv2.imread(fname_in)

    # im = skdata.camera()
    # background_gradient = np.zeros_like(im)

    # for i in range(len(background_gradient)):
    #    background_gradient[i] += np.arange(0., 50., 50/len(background_gradient[i]))

    _use_cython = False
    _use_cpp = True
    w_str = "Processing"
    invert = True
    factor = 1
    if invert:
        factor = -1

    im_hsv = cv2.cvtColor(im[:, :, :3], cv2.COLOR_RGB2HSV)
    v_im = factor * im_hsv[:, :, 2]
    # v_im = factor * cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    im_orig = np.copy(factor * v_im)
    res_factor = 1

    if res_factor > 1:
        im_dwn = _shrink_image(v_im, res_factor)
    else:
        im_dwn = v_im

    radius = 20
    pixels = im_dwn.flatten()
    w = im_dwn.shape[1]
    h = im_dwn.shape[0]
    if _use_cython:
        cp_par.sliding_paraboloid_float_background(pixels.tolist(), w, h, radius, False, False)

    elif _use_cpp:
        lib = ctypes.cdll.LoadLibrary('cpp_slide_paraboloid.so')
        fun = lib.sliding_paraboloid_float_background
        fun.restype = None
        fpixels = pixels.astype(np.float32)
        fun.argtypes = [ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool, ctypes.c_bool]

        fun(fpixels, w, h, radius, False, False)
        pixels = fpixels.astype(pixels.dtype)
    else:
        sliding_paraboloid_float_background(pixels, w, h, radius, False, False)

    im_dwn[:] = np.array(pixels).reshape((h, w))[:]

    if res_factor > 1:
        im_up = (factor * _enlarge_image(im_dwn, res_factor)).astype(np.uint8)
    else:
        im_up = (factor * im_dwn).astype(np.uint8)

    im_res = np.zeros_like(im_orig)
    im_res = im_orig - im_up + 255.5
    im_res = np.clip(im_res, 0, 255).astype(np.uint8)

    cv2.namedWindow(w_str, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(w_str, 900, 300)
    cv2.imshow(w_str, np.hstack([im_orig.astype(np.uint8), im_up, im_res]))

    res_hsv = im_hsv[:]
    res_hsv[:, :, 2] = im_res[:]

    res_im = cv2.cvtColor(res_hsv, cv2.COLOR_HSV2RGB)

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 600, 300)
    cv2.imshow("Result", np.hstack([im, res_im]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(fname_out, res_im)
