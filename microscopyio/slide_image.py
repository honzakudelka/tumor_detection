import os

import cv2
import numpy as np

from microscopyio import hp_tiff as tiff
from microscopyio import ndpi


class SlideImage(object):
    """
    Class SlideImage implements a base class needed for handling Whole-slide images (WSI). Currently, the
    .ndpi (PETACC Trial Data) and .tiff (CAMELYON data) formats are supported.

    :image_path Path to the image
    :tissue_mask_path Tissue mask with 0 - background and >0 in tissue pixels
    :tumor_mask_path Path to the tumor mask
    :tumor_annotation_file Path to an XML description of the tumor area

    Should the labels be considered, either tumor_mask_path or the tumor_annotation_file are required to be set

    The class first reads the metadata (number of different resolution, scaling factor for each resolution) and the
    tumor mask or tumor annotation file if provided. Image data are loaded on demand via the load_patch routine.
    """

    def __init__(self,
                 image_path, tissue_mask_path,
                 tumor_mask_path=None, tumor_annotation_file=None):
        self._path = image_path
        self._tissue_mask = tissue_mask_path
        self._tumor_mask = tumor_mask_path
        self._annotation_file = tumor_annotation_file

        self._slide_img = None
        self._np_img = None
        self._np_mask = None
        self._np_tmask = None
        self._image_type = None
        self._automask_patch = False

        self._img_levels = None
        self._img_dimensions = None
        self._img_downsamples = None
        self._probe_lebel = 0

        self._min_cnt_area = 600
        self._visualize = False

    @staticmethod
    def _show_image(image, w_name):

        cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(w_name, 900, 900)
        cv2.imshow(w_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _load_image_meta(self):
        """
        Load meta-data from the given whole slide image

        :return: None
        """
        raise NotImplementedError("The SlideImage base class cannot load images, use one of the derived classes.")

    def load_patch(self, position, size, level=0):
        """
        Load a patch from the image at specified position and resolution level (0 is the maximal available resolution)

        :param position: Tuple of 2 elements defining the extract position
        :param level: Level for extraction
        :param size: Size of the patch to be extracted. Provide either a single unsigned int (for square patches) or a tuple of 2 unsigned ints for
        general rectangular patches
        :return: numpy array (RGB) containing the patch
        """

        if self._slide_img is None:
            self._load_image_meta()

        if isinstance(size, tuple):
            p_size = (int(size[0] / self._img_downsamples[level]),
                      int(size[1] / self._img_downsamples[level]))
        else:
            p_size = int(size / self._img_downsamples[level])

        patch = self._slide_img.read_region(location=position, level=level, size=p_size)
        patch = cv2.cvtColor(np.array(patch), cv2.COLOR_RGBA2RGB)

        if self._visualize:
            self._show_image(patch,
                             "Extracted patch at {},{}".format(position[0], position[1]))

        return patch

    def _compute_patch_cover(self, probe_level=0, patch_size=1024, patch_shift=1024, min_coverage=0.95):
        """
        Compute the covering mosaic of rectangular patches of given size

        :param probe_level: the zooming level to work on
        :param patch_size: size (one axis) of the (rectangular) patch, always the size at level 0
        :param patch_shift: the shift between two neighboring patches, default value is the patch size (i.e. no overlap)
        :param min_coverage: minimum tissue coverage required for a patch to be included
        """

        if self._slide_img is None:
            self._load_image_meta()

        extract_level = min(probe_level, self._img_levels)
        np_img = self._slide_img.read_region((0, 0),
                                             extract_level,
                                             size=self._slide_img.level_dimensions[extract_level])
        self._np_img = cv2.cvtColor(np.array(np_img), cv2.COLOR_RGBA2RGB)

        np_mask = cv2.imread(self._tissue_mask, cv2.IMREAD_GRAYSCALE)
        if np_mask is None:
            raise RuntimeError("Mask file {} could not be loaded".format(self._tissue_mask))

        self._np_mask = cv2.resize(np_mask, (self._np_img.shape[1], self._np_img.shape[0]), cv2.INTER_NEAREST)

        tissue = np.unique(self._np_mask)[-1]
        thr, mask_thr = cv2.threshold(self._np_mask, tissue - 1, 255, cv2.THRESH_BINARY)
        contours, c_hier = cv2.findContours(mask_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        informed_contours = [(cnt, cv2.contourArea(cnt)) for cnt in contours]

        probe_size_x = int(patch_size / (self._img_downsamples[extract_level]))
        probe_shift_x = int(patch_shift / (self._img_downsamples[extract_level]))
        if self._visualize:
            probe_image_coverage = self._np_img.copy()

        patches = []
        for cnt_i, contour in enumerate(informed_contours):

            coverage_im = np.zeros_like(mask_thr)

            if contour[1] < self._min_cnt_area:
                continue

            x, y, w, h = cv2.boundingRect(contour[0])
            cv2.drawContours(coverage_im, [contour[0]], 0, 255, -1)
            coverage_im = cv2.erode(coverage_im, (3, 3))
            if self._visualize:
                print("Extracting from bbox {4},{5}+{0}x{1}, contour area: {2} at level {3}".format(
                    w, h,
                    contour[1],
                    extract_level,
                    x, y
                ))
                cv2.drawContours(self._np_mask, [contour[0]], 0, 0, 2, cv2.LINE_AA)
                # cv2.rectangle(self._np_mask, (x, y), (x + w, y + h), 0, 1, cv2.LINE_AA)
                self._show_image(self._np_mask, "Patch grid")

            if min(w, h) < probe_size_x:
                print("Not enough pixels in contour {}".format(cnt_i))
                continue

            print(" Probe image: {}x{} - {}x{}".format(x, y, w, h))

            for i in range(int(np.ceil(1. * w / probe_shift_x))):
                for j in range(int(np.ceil(1. * h / probe_shift_x))):
                    px = x + i * probe_shift_x
                    py = y + j * probe_shift_x
                    pos = (int(px * self._img_downsamples[probe_level]),
                           int(py * self._img_downsamples[probe_level]))

                    print(" >> patch:e image: {}x{} + {}x{}".format(px, py, probe_size_x, probe_size_x))
                    patch_raw = self._np_img[py:py + probe_size_x, px:px + probe_size_x]
                    cnt_val = coverage_im[py:py + probe_size_x, px:px + probe_size_x]
                    mask_val = mask_thr[py:py + probe_size_x, px:px + probe_size_x]

                    if patch_raw.size < 3 * probe_size_x ** 2:
                        continue

                    coverage = min(sum(mask_val.reshape((1, probe_size_x ** 2))[0]),
                                   sum(cnt_val.reshape((1, probe_size_x ** 2))[0]))

                    sp_rect_color = (0, 0, 255)
                    if coverage / 255.0 > min_coverage * probe_size_x ** 2:
                        patches.append(pos)
                        sp_rect_color = (0, 255, 0)

                    if self._visualize and coverage > 0:
                        cv2.rectangle(probe_image_coverage,
                                      (px, py),
                                      (px + probe_size_x, py + probe_size_x), sp_rect_color, 1, cv2.LINE_AA)

        if self._visualize:
            cv2.namedWindow("Patch Cover", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Patch Cover", 900, 900)
            cv2.imshow("Patch Cover", probe_image_coverage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return patches

    def _load_annotations(self, extract_level=6):
        """
        Load annotation files, use either ndpi.read_annotation_file or tiff.read_annotation_file functions

        :param extract_level: Scale annotation to fit the zooming level
        :return: A list of annotations (each annotation is a list of contour points)
        """
        raise NotImplementedError("Attempting to call base class' method, use the image-specific classes instead.")

    def _get_tumor_mask(self, extract_level=6, save_prefix=None):
        """
        Internal method to get the tumor mask

        :param extract_level:
        :param save_prefix:
        :return:
        """

        if self._slide_img is None:
            self._load_image_meta()

        # retrieve annotation
        annotations = self._load_annotations(extract_level)

        image_size = self._img_dimensions[extract_level]
        # account for OpenCV loading the image in reverse axis order
        annot_mask = np.zeros((image_size[1], image_size[0]), dtype="uint8")

        if self._np_img is None:
            np_img = self._slide_img.read_region((0, 0),
                                                 extract_level,
                                                 size=self._slide_img.level_dimensions[extract_level])
            self._np_img = cv2.cvtColor(np.array(np_img), cv2.COLOR_RGBA2RGB)

        background_img = self._np_img.copy()
        for k in annotations:
            class_fill = 255

            if '[_2]' in k:
                class_fill = 0

            cv2.drawContours(annot_mask, [annotations[k]], 0, class_fill, -1)
            cv2.drawContours(background_img, [annotations[k]], 0, (255, 0, 255), 3, cv2.LINE_AA)

        if save_prefix is not None:
            cv2.imwrite(save_prefix + "_l{}_Annot.png".format(extract_level), background_img)
            cv2.imwrite(save_prefix + "_l{}_Mask.png".format(extract_level), annot_mask)

        if self._visualize:
            cv2.namedWindow("Tumor mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tumor mask", 900, 900)
            cv2.imshow("Tumor mask", np.hstack([#cv2.cvtColor(annot_mask, cv2.COLOR_GRAY2RGB),
                                                background_img]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annot_mask

    def get_patches(self, extract_level, p_size, min_coverage=0.75):
        """
        Get patch cover of the tissue class for a slide image. The instance must be provided with a image_path and tissue_mask_path
        prior to calling this method

        :param extract_level: Zooming level for extraction, image data are extracted from the given slide image, mask image
        is resampled to fit this size
        :param p_size: Size of the patch (rectangular of size p_size x p_size)
        :param min_coverage: Minimal tissue coverage within patch (default=0.75)
        :return: list containing positions (upper left corner coordinates) of valid patches
        """
        return self._compute_patch_cover(extract_level, patch_size=p_size, min_coverage=min_coverage)

    def get_annotated_patches(self, extract_level, p_size, min_coverage_extraction, min_tumor_coverage, p_shift):
        """Retrieve complete list of patches together with their annotation labels. The annotation file is either at the
        default location or provided prior to method call by setting the self._annotation_file member variable to point to
        the required file.

        :param extract_level: Zooming level for extraction
        :param p_size: patch size (for zooming level 0)
        :param p_shift: offset for the next patch (if less then p_size, the patches will overlap)
        :param min_coverage_extraction: minimal tissue coverage for patch to be included
        :param min_tumor_coverage: minimal tumor tissue coverage to get 'TU' label, other patches retrieve either 'BO' or
        'NO' labels for normal
        :return:
        """

        all_patches = self._compute_patch_cover(extract_level, p_size, p_shift, min_coverage_extraction)
        try:
            annotation_mask = self._get_tumor_mask(extract_level)

        except IOError as e:
            print("Caught IO Exception during reading tumor annotations. \n " + str(e))
            annotation_mask = np.zeros(self._np_img.shape[:2])

        self._tumor_mask = annotation_mask
        print("Annotation mask:", annotation_mask.shape)
        print("Image size:", self._np_img.shape)

        # initialize empty list
        ann_patches = [[], [], []]
        # patch size at current level
        probe_size_x = int(p_size / self._img_downsamples[extract_level])
        # pixel_value / patch_area
        inv_patch_area = 1. / (255. * probe_size_x * probe_size_x)
        counts = [0, 0, 0]
        for patch in all_patches:
            px = int(patch[1] / self._img_downsamples[extract_level])
            py = int(patch[0] / self._img_downsamples[extract_level])
            mask_coverage = inv_patch_area * np.sum(annotation_mask[px:px + probe_size_x, py:py + probe_size_x])

            if min_tumor_coverage <= mask_coverage:
                label = 'TU'
                ann_patches[0].append((patch, label))
            elif 0.1 < mask_coverage < min_tumor_coverage:
                label = 'BO'
                ann_patches[1].append((patch, label))
            else:
                label = 'NO'
                ann_patches[2].append((patch, label))

        print("Got {} tumor, {} border and {} normal patches".format(
            len(ann_patches[0]), len(ann_patches[1]), len(ann_patches[2]))
        )

        # return ordered by label
        return ann_patches[0] + ann_patches[1] + ann_patches[2]

    def get_patch_visualization(self, display_level, patch_list, patch_size, line_thickness=1, scalars=None,
                                border_scalars=None,
                                show=False, filled=False, shrink=0, output_map=None,
                                offset=0, scalar_scale=1.0):
        """Draw all patches over the slide image and return an numpy array

        :param display_level: zooming level of the image to display
        :param patch_list: list of patches to be drawn
        :param patch_size: size of patch for level 0
        :param show: boolean flag to specify whether to directly visualize the image or only return it
        :return: RGB image (as 2D numpy array) containing the original slide image overlaid with the patch mosaic
        """

        if self._np_img is None:
            self._load_image_meta()
            np_img = self._slide_img.read_region((0, 0),
                                                 display_level,
                                                 size=self._slide_img.level_dimensions[display_level])
            self._np_img = cv2.cvtColor(np.array(np_img), cv2.COLOR_RGBA2RGB)

        background_img = self._np_img.copy()
        if scalars is not None:
            background_img = cv2.cvtColor(self._np_img.copy(), cv2.COLOR_RGB2RGBA)
        fill_overlay = 255 * np.ones_like(background_img, dtype=np.uint8)

        color_scheme = {'TU': (0, 0, 255), 'BO': (0, 255, 255), 'NO': (0, 255, 0)}
        scale_factor = 1. / self._img_downsamples[display_level]
        display_patch_size = int(patch_size * scale_factor)

        color_image = None
        if scalars is not None:
            probe_image = np.arange(0, 255, 2, 'uint8')
            color_image = cv2.applyColorMap(probe_image, cv2.COLORMAP_JET)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2RGBA)

            if border_scalars is None:
                border_scalars = np.copy(scalars)

        output_img = None
        output_mask = None
        if output_map is not None:
            output_img = np.zeros((background_img.shape[0], background_img.shape[1]), dtype=np.float)
            output_mask = np.zeros_like(output_img)

        for p_i, patch in enumerate(patch_list):
            patch_pos = patch[0]
            patch_label = patch[1]

            px = int((patch_pos[0]) * scale_factor)
            py = int((patch_pos[1]) * scale_factor)

            p_color = color_scheme[patch_label]

            if scalars is not None:
                if patch_label == 'TU':
                    p_color = (0, 0, 255, 255)
                else:
                    p_color = (0, 255, 0, 255)

                f_color = color_image[int(scalars[p_i])].tolist()[0]

            draw = True
            lt = line_thickness

            if lt < 0:
                draw = False
            else:
                if border_scalars is not None:  # and scalars[p_i] <= scalar_min:
                    if (patch_label == 'TU' or patch_label == 'BO') and border_scalars[p_i] > 50:
                        draw = False
                    if patch_label == 'NO' and border_scalars[p_i] < 50:
                        draw = False

                    if draw and (border_scalars[p_i] < 30 or border_scalars[p_i] > 70):
                        lt = 2 * line_thickness

            if draw:
                cv2.rectangle(background_img,
                              (px + shrink, py + shrink),
                              (px + display_patch_size - shrink, py + display_patch_size - shrink), p_color, lt,
                              cv2.LINE_AA)

            if filled:
                cv2.rectangle(fill_overlay,
                              (px + shrink, py + shrink),
                              (px + display_patch_size - 2 * shrink, py + display_patch_size - 2 * shrink), f_color,
                              cv2.FILLED, cv2.LINE_AA)
            if output_img is not None:
                patch_result = scalars[p_i] * np.ones((display_patch_size, display_patch_size), dtype=float)
                output_img[py:py + display_patch_size, px:px + display_patch_size] += offset + scalar_scale * patch_result
                output_mask[py:py + display_patch_size, px:px + display_patch_size] += 1

        if show:
            cv2.namedWindow("Patch Cover", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Patch Cover", 900, 900)
            cv2.imshow("Patch Cover", background_img)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()

        if filled:
            alpha = .75
            background_img = cv2.cvtColor(
                cv2.addWeighted(background_img, alpha, fill_overlay, 1 - alpha, 0),
                cv2.COLOR_RGBA2RGB
            )

            for i in range(100):
                f_color = color_image[100 - i - 1].tolist()[0]
                f_color = f_color[:3]

                cv2.rectangle(background_img,
                              (background_img.shape[1] - 2 * display_patch_size, 10 + i * 5),
                              (background_img.shape[1] - display_patch_size, 15 + i * 5), f_color, cv2.FILLED,
                              cv2.LINE_AA
                              )

                if (100 - i) % 10 == 0:
                    cv2.line(background_img,
                             (background_img.shape[1] - 3 * display_patch_size, 15 + i * 5),
                             (background_img.shape[1] - display_patch_size, 15 + i * 5), (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(background_img, '{:0.1f}'.format((100 - i) * 0.01),
                                (background_img.shape[1] - 4 * display_patch_size, 15 + i * 5),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1, cv2.LINE_AA
                                )

        if output_map is not None:
            out_max = np.max(output_img)
            output_img[np.where(output_mask < 1)] = 0
            output_map.append(output_img)

            patch_score = []
            for p_i, patch in enumerate(patch_list):
                patch_pos = patch[0]

                px = int((patch_pos[0]) * scale_factor)
                py = int((patch_pos[1]) * scale_factor)

                output_img[py:py + display_patch_size, px:px + display_patch_size] /= output_mask[
                    py + display_patch_size // 2, px + display_patch_size // 2]

                patch_score.append(np.mean(output_img[py:py + display_patch_size, px:px + display_patch_size]))

            output_map.append(patch_score)

        return background_img


class CamelyonSlideImage(SlideImage):
    """SlideImage class for CAMELYON'16 data (and associated XML annotations)
    """

    def __init__(self, image_path, tissue_mask_path, tumor_mask_file=None, tumor_annotation_file=None):
        super(CamelyonSlideImage, self).__init__(image_path=image_path, tissue_mask_path=tissue_mask_path,
                                                 tumor_mask_path=tumor_mask_file,
                                                 tumor_annotation_file=tumor_annotation_file)
        self._image_type = "CAMELYON"

    def _load_image_meta(self):
        self._slide_img = tiff.read_file(self._path)
        if self._slide_img is None:
            raise RuntimeError("File " + self._path + " not found!")

        self._img_dimensions = self._slide_img.level_dimensions[:]
        self._img_levels = self._slide_img.level_count
        # camelyon'16 hack, downsampling levels are 2**n, don't take the stored information in the image
        self._img_downsamples = [2 ** i for i in range(self._slide_img.level_count)]
        self._img_orig_downsamples = self._slide_img.level_downsamples

    def _load_annotations(self, extract_level=6):
        if self._annotation_file is None:
            annotation_root = "/mnt/medical/microscopy/CAMELYON16/Train-Ground_Truth/XML2"
            self._annotation_file = annotation_root + "/" + os.path.splitext(os.path.basename(self._path))[0] + ".xml"

        if not os.path.exists(self._annotation_file):
            raise IOError("Annotation file {} does not exist.".format(self._annotation_file))

        ann_size = self._img_downsamples[extract_level]
        return tiff.read_annotation_file(self._annotation_file,
                                         (ann_size, ann_size))

    def load_patch(self, position, size, level=0):
        """

        :param position: Tuple of 2 elements defining the extract position
        :param level: Level for extraction
        :param size: Size of the patch to be extracted. Provide either a single unsigned int (for square patches)
        or a tuple of 2 unsigned ints for general rectangular patches
        :return: numpy array (RGB) containing the patch
        """

        if self._slide_img is None:
            self._load_image_meta()

        if isinstance(size, tuple):
            p_size = (int(size[0] / self._img_downsamples[level]),
                      int(size[1] / self._img_downsamples[level]))
        else:
            p_size = (int(size / self._img_downsamples[level]),
                      int(size / self._img_downsamples[level]))

        drift = np.zeros((2,), dtype=int)
        drift[0] = self._img_dimensions[0][0] - self._img_dimensions[level][0] * self._img_orig_downsamples[level]
        drift[1] = self._img_dimensions[0][1] - self._img_dimensions[level][1] * self._img_orig_downsamples[level]

        factor_drift = max(self._img_dimensions[0][0] / (1. * self._img_dimensions[level][0]),
                           self._img_dimensions[0][1] / (1. * self._img_dimensions[level][1])) / \
                       self._img_orig_downsamples[level]

        factor_drift += 1e-5 * (0.5 * abs(drift[0] + drift[1]))

        # print("cam16  DRIFT: " + str(drift))
        # print("factor DRIFT: " + str(factor_drift))

        #extract_position = (int(position[0] / factor_drift), int(position[1] / factor_drift))
        extract_position = (int(position[0]), int(position[1]))

        patch = self._slide_img.read_region(location=extract_position, level=level, size=p_size)
        patch = cv2.cvtColor(np.array(patch), cv2.COLOR_RGBA2RGB)

        if self._visualize:
            self._show_image(patch,
                             "Extracted patch at {},{}".format(extract_position[0], extract_position[1]))

        return patch

    def get_annotated_normal_patches(self, extract_level, p_size, min_coverage_extraction):
        """Retrieve all patches and decorate them with the 'NO' label. Assumption - the file is a normal (healthy) sample
        without tumor annotation."""

        all_patches = self._compute_patch_cover(extract_level, p_size, min_coverage_extraction)
        ann_patches = []
        for patch in all_patches:
            label = 'NO'
            ann_patches.append((patch, label))

        return ann_patches


class NDPISlideImage(SlideImage):
    """SlideImage class specification for NDPI data (NDPA or XML annotations)

    """

    def __init__(self, image_path, tissue_mask_path, tumor_mask_file=None, tumor_annotation_file=None):
        super(NDPISlideImage, self).__init__(image_path=image_path,
                                             tissue_mask_path=tissue_mask_path,
                                             tumor_mask_path=tumor_mask_file,
                                             tumor_annotation_file=tumor_annotation_file)

        self._image_type = "NDPI"
        self._ndpi_metadata = None

    def _load_image_meta(self):

        self._slide_img, self._ndpi_metadata = ndpi.read_file(self._path)
        self._img_dimensions = list(self._ndpi_metadata.LevelDimensions[:])
        self._img_downsamples = self._ndpi_metadata.DownsamplingFactor
        self._img_levels = self._slide_img.level_count

    def _load_annotations(self, extract_level=6):
        if self._annotation_file is None:
            if os.path.exists(self._path + ".ndpa"):
                self._annotation_file = self._path + ".ndpa"
            elif os.path.exists(self._path + ".xml"):
                self._annotation_file = self._path + ".xml"
            else:
                raise RuntimeError("No annotation file specified and no file detected on standard locations.")

        return ndpi.read_annotation_file(self._annotation_file,
                                         self._ndpi_metadata,
                                         extract_level)

    def load_patch(self, position, size, level=0):
        """
        Loads patch at full resolution.

        If the input data contain some manual markings on the slide,
        turn on the self._automask_patch flag and the method will estimate such pixels from the upsampled
        mask image and set those with non-tissue label to value (255, 255, 255)

        :param position: patch position
        :param size: patch size
        :return:
        """

        patch = super(NDPISlideImage, self).load_patch(position, size, level)

        # we can try to automask the patch, there is a chance, that it still
        # contains some of the marker lines (applies only to petacc data)
        if self._automask_patch and self._np_mask is not None:
            # detect level
            fx = 1.0 * self._img_dimensions[0][1] / self._np_mask.shape[0]
            fy = 1.0 * self._img_dimensions[0][0] / self._np_mask.shape[1]

            x_index = self._img_downsamples.index(fx)
            y_index = self._img_downsamples.index(fy)

            tissue_val = np.max(self._np_mask)

            px = int(position[1] / fx)
            py = int(position[0] / fy)
            w = int(p_size[1] / fx)
            h = int(p_size[0] / fy)

            patch_tissue_mask = self._np_mask[px:px + w, py:py + h]
            mask = np.ones_like(patch_tissue_mask)
            mask[np.where(patch_tissue_mask < tissue_val)] = 0

            mask = cv2.resize(mask, p_size, cv2.INTER_NEAREST)

            orig_patch = patch.copy()

            if x_index == y_index:
                # mask tumor
                replacement = (255, 255, 255)
                patch[np.where(mask < 1)] = replacement
            else:
                print("Cannot deduce mask extraction level.")

        # if self._visualize:
        #     self._show_image(np.hstack([patch, orig_patch]),
        #                      "Extracted patch at {},{}".format(position[0], position[1]))

        return patch
