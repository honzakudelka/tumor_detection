import os

import numpy as np
from tflearn.data_utils import Preloader, LabelPreloader

from microscopyio.slide_image import NDPISlideImage, CamelyonSlideImage


class MicroscopyPreloader(Preloader):
    """
    Preloader (inherits tflearn.data_utils.Preloader) for microscopy data, which can be used
    as input for a CNN within tflearn framework.

    When creating a preloader, a meta-data array of form (image_path, patch_location, patch_size),
    the microscopy image type (default=ndpi) and the zooming level for extraction (default = 2)
    must be specified.

    When passed to TFLearn

    """

    def __init__(self, meta_data_array, mic_type='ndpi',
                 extract_level=2):
        def fn(x):
            return self.preload(x)

        super(MicroscopyPreloader, self).__init__(meta_data_array, fn)

        if mic_type == 'ndpi':
            self._image_type = NDPISlideImage
        elif mic_type == 'tiff':
            self._image_type = CamelyonSlideImage
        else:
            raise RuntimeError("Unsupported microscopy image type {}".format(mic_type))

        self.extract_level = extract_level
        self.last_name = None
        self.si = None

    def preload(self, patch_meta):

        if self.last_name is None or not self.last_name == patch_meta[0]:
            self.si = self._image_type(patch_meta[0], None)
            self.last_name = patch_meta[0]

        img = self.si.load_patch(patch_meta[1], patch_meta[2], self.extract_level)
        return np.array(img, dtype=np.float32)


def microscopy_preloader(meta_data_list, extract_level, extract_size, original_size=(1024, 1024),
                         n_classes=2, label_dictionary=None, labels=None):
    """
    Create a preloader (TFLearn style) for PETACC dataset.

    :param meta_data_list: a tuple with (microscopy image path, list of patch positions, extraction size)
    :param extract_level:  zooming level for extraction
    :param extract_size:  patches of this size will be extracted
    :param original_size: size of the patches in the metadata array
    :param n_classes: number of classes ( 2- binary classification)
    :param label_dictionary: dictionary for translating labels in meta-data to true labels
    :param labels:  patch labels    - either label_dictionary or labels *must* be set
    :return:
    """

    if len(meta_data_list) < 1:
        raise RuntimeError("Empty meta data list given")

    if label_dictionary is None and labels is None:
        raise RuntimeError("Specify either label_dictionary or provide list of labels!")

    probe_image = meta_data_list[0]
    if probe_image[0].find('.ndpi'):
        loader_type = 'ndpi'
    else:
        loader_type = 'tiff'

    # Estimate number of sub-sampling steps, when original size (used in the metadata) is a multiple of the
    # requested extract size
    subsamples = original_size[0] // extract_size[0]

    all_images_meta = []
    all_images_labels = []
    petacc_root = "/datagrid/Medical/microscopy/petacc3"

    path_candidates = ['batch_2', 'batch_1']

    for wsi_meta in meta_data_list:
        image_name = os.path.basename(wsi_meta[0])[:8]

        image_file_found = False
        for subdir in path_candidates:
            image_path = petacc_root + "/{}/".format(subdir) + image_name + ".ndpi"

            if os.path.exists(image_path):
                image_file_found = True
                break

        if not image_file_found:
            raise RuntimeError("File {} not found in any candidate paths...".format(image_name))

        for pi, patch in enumerate(wsi_meta[2]):

            patch_position = np.array([patch[0][0], patch[0][1]])

            if label_dictionary is not None:
                label = label_dictionary[patch[1]]
            else:
                label = labels[pi]

            for i in range(subsamples):
                for j in range(subsamples):
                    subsampled_position = (
                    patch_position[0] + i * extract_size[0], patch_position[1] + j * extract_size[1])

                    all_images_meta.append((image_path, subsampled_position, extract_size))
                    all_images_labels.append(label)

    X = MicroscopyPreloader(all_images_meta, loader_type, extract_level)
    Y = LabelPreloader(all_images_labels, n_classes, True)

    return X, Y
