import argparse
import numpy as np
import os

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Process prediction maps")

    parser.add_argument("--input", type=str, help="Input (prediction map)",
                        required=True)
    parser.add_argument("--label", type=str, help="Image label (NO or TU)",
                        required=True)
    parser.add_argument("--lmap", type=str, help="Annotation (needed only for TU labeled images)",
                        required=False)
    parser.add_argument("--bmap", type=str, help="Path to bluriness map image", required=True)

    args = parser.parse_args()

    im_factor = 0.01
    scale_factor = 0.25

    pmap = np.clip(im_factor * 2 * (cv2.imread(args.input, cv2.IMREAD_GRAYSCALE) - 20.), 0.0, 1.0)
    pmap_im = np.zeros((int(scale_factor * pmap.shape[0]), int(pmap.shape[1] * scale_factor)), dtype=np.float)
    cv2.resize(pmap, dst=pmap_im, dsize=None, fx=scale_factor, fy=scale_factor,
               interpolation=cv2.INTER_CUBIC)

    bmap = np.clip(im_factor * (cv2.imread(args.bmap, cv2.IMREAD_GRAYSCALE)), 0.0, 1.0)
    bmap_im = np.zeros((int(scale_factor * bmap.shape[0]), int(bmap.shape[1] * scale_factor)), dtype=np.float)
    cv2.resize(bmap, dst=bmap_im, dsize=None, fx=scale_factor, fy=scale_factor,
               interpolation=cv2.INTER_CUBIC)

    bgmap_path = args.bmap.replace('blur.png','bckg.png')
    bgmap = np.clip(im_factor * (cv2.imread(bgmap_path, cv2.IMREAD_GRAYSCALE)), 0.0, 1.0)
    bgmap_im = np.zeros((int(scale_factor * bgmap.shape[0]), int(bgmap.shape[1] * scale_factor)), dtype=np.float)
    cv2.resize(bgmap, dst=bgmap_im, dsize=None, fx=scale_factor, fy=scale_factor,
               interpolation=cv2.INTER_CUBIC)

    # w_name = "Map compare"
    # cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(w_name, 800, 400)
    # cv2.imshow(w_name, np.hstack([pmap_im, bmap_im, np.minimum(pmap_im, np.abs(pmap_im - bmap_im))]))
    # cv2.waitKey(0)

    x_pts = []
    y_pts = []
    z_pts = []
    for y in range(bmap_im.shape[0]):
        for x in range(bmap_im.shape[1]):
            if pmap_im[y][x] < 1e-3:
                continue

            x_pts.append(bmap_im[y][x])
            y_pts.append(pmap_im[y][x])
            z_pts.append(bgmap_im[y][x])

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.scatter(x_pts, y_pts,  s=10, facecolors='b', edgecolors='b')
    ax1.set_xlabel("Blur score")
    ax1.set_ylabel("Prediction score")
    ax1.set_title("Blur score")

    ax2.scatter(z_pts, y_pts,  s=10, facecolors='r', edgecolors='r')
    ax2.set_xlabel("Background score")
    ax2.set_ylabel("Prediction score")
    ax2.set_title("Background score")

    plt.show()




