from feature_extraction import textural
import cv2
import numpy as np
from openslide import OpenSlide

if __name__ == "__main__":
    img_id = '011'

    study_training_root = "/mnt/medical/microscopy/CAMELYON16/training"
    study_testing_root = "/mnt/medical/microscopy/CAMELYON16/testing"

    normal_path = "/normal/Normal_{0}.tif".format(img_id)
    tumor_path = "/tumor/tumor_{0}.tif".format(img_id)
    test_path = "/Test_{0}.tif".format(img_id)

    path = "/mnt/medical/microscopy/CAMELYON16/training/normal/Normal_135.tif"

    location = (18176, 66560)
    level = 0
    size = (1024, 1024)
    visualize = True

    img = OpenSlide(path)
    patch = img.read_region(location=location, level=0, size=(1024, 1024))
    patch = np.array(patch)

    patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2BGR)
    grey_norm = np.zeros_like(patch)
    grey_norm = cv2.normalize(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), dst=grey_norm,
                              alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    textural_features = textural.compute_textural_features(grey_norm)

    for k, v in textural_features.items():
        print("[TX-FEATURE] {0} : {1}".format(k, v))
