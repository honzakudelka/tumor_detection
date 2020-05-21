import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

import numpy as np
from openslide import OpenSlide, OpenSlideError


def read_file(path):
    """
    Reads a TIF image through OpenSlide library

    :param path: Path to input file (TIF)
    :return: OpenSlide Image Object or Exception if invalid path
    """

    tiff_image = None
    try:
        tiff_image = OpenSlide(path)
    except OpenSlideError as oe:
        print("[OpenSlideException] " + str(oe))
    except IOError as e:
        print("Failed to read data, \n [IO-Exception] {0}".format(str(e)))

    return tiff_image


def read_annotation_file(annonfile, downsampling_factor):
    """
    Reads an annotation file (for XML format from CAMELYON challenge)

    The XML file contains possibly multiple Annotations, this methods reads them all, stores them in a dictionary. The key
    is composed of the [Name] and the [PartOfGroup] properties of the annotations. _1, _0 tumor areas, _2 non-tumor areas enclosed by tumor areas,
    all remaining is healthy tissue.




    :param annonfile: Path to XML file with annotations
    :param downsampling_factor: important to scale the coordinates correctly for the image

    :return: a dictionary with annotations - [key] is the name and [value] a point list with coordinates (downsampled with the factor)
    """
    xml_file = ET.parse(annonfile)
    annotations = {}

    an = xml_file.find('Annotations')
    for annotation in list(an):
        clist = annotation.find('Coordinates')
        if clist is None:
            continue

        xy_coords = []
        for points in list(clist):
            x = float(points.get('X'))
            y = float(points.get('Y'))

            x_px = x / downsampling_factor[0]
            y_px = y / downsampling_factor[1]

            xy_coords.append([x_px, y_px])

        if len(xy_coords) < 5:
            # too short
            continue

        # check the last point to match the first one
        if (xy_coords[0][0] != xy_coords[-1][0]) or (xy_coords[0][1] != xy_coords[-1][1]):
            xy_coords.append(xy_coords[0])

        xy_coords = np.array(xy_coords, dtype=np.int32)

        ann_name = "{0} [{1}]".format(annotation.get('Name'),
                                      annotation.get('PartOfGroup'))

        annotations.update({ann_name: xy_coords})

    return annotations


def read_image(file, level=0, annon_root=None):
    """
    Reads a microscopy, multi-level TIF image with OpenSlide library and returns a thumbnail as PIL image
    at the given zooming level.

    If provided, also annotations are read in and displayed as overlay. If a _Mask.tif annotation is found, it is used instead
    of the XML annotation. If only XML is provided, a mask is computed based on the polygons
    ( 0 - default, 127 for tumor, and 255 for non-tumor enclosed by tumor)


    :param file: Path to the image file in TIF format
    :param level: Specify at which level the image should be extracted (0-best resolution, MAX-lowest)
    :param annon_root: Path to the directory with annotations (expected names for CAMELYON <TIF_FILE_NAME>.xml,
                       or None (default) if no annotations should be loaded

    :return: a tuple of ( raw image, image with annotation overlay, annotation mask) all three values are PIL images and
             and have identical resolution (based on the selected zoom level)
    """

    tiff_image = read_file(file)
    _level = min(level, tiff_image.level_count - 1)

    annotations = None
    maskimage = None

    if annon_root is not None:
        annon_file = annon_root + "/" + os.path.splitext(os.path.basename(file))[0] + ".xml"
        mask_file = annon_root + "/" + os.path.splitext(os.path.basename(file))[0] + "_Mask.tif"

        if os.path.exists(annon_file):
            xyfac = tiff_image.level_dimensions[_level]
            # necessary hack for CAMELYON|16 data, the original levels does not fit and create an offset
            ds_x = pow(2, _level)  # tiff_image.level_dimensions[0][0] / (1.0 * xyfac[0])
            ds_y = pow(2, _level)  # tiff_image.level_dimensions[0][1] / (1.0 * xyfac[1])
            annotations = read_annotation_file(annon_file, (ds_x, ds_y))

        if os.path.exists(mask_file):
            maskimage = read_file(mask_file)
            if maskimage.level_count != tiff_image.level_count:
                raise RuntimeError('Different number of resolution levels in image / mask')

    draw_annot = False
    overlay_mask = False

    # Prioritize mask over annotation
    if maskimage is not None:
        overlay_mask = True
    elif annotations is not None:
        draw_annot = True

    image = tiff_image.get_thumbnail(tiff_image.level_dimensions[_level]).convert('RGBA')
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    annot_mask = Image.new('L', image.size, 0)

    d = ImageDraw.Draw(overlay)
    da = ImageDraw.Draw(annot_mask)

    if overlay_mask:
        mask_img = maskimage.get_thumbnail(image.size).convert('RGBA')
        overlay = mask_img

    if draw_annot:
        for k in annotations:
            vert_list = []

            outlinec = (0, 0, 255, 128)
            class_fill_8bit = 127
            if '[_2]' in k:
                outlinec = (0, 255, 0, 128)
                class_fill_8bit = 255

            for i, point in enumerate(annotations[k]):

                start = point
                vert_list.append((start[0], start[1]))

                if i < len(annotations[k]) - 1:
                    end = annotations[k][i + 1]
                else:
                    continue

                d.line((start[0], start[1], end[0], end[1]), fill=outlinec, width=8)

            da.polygon(vert_list, fill=class_fill_8bit)
            # d.polygon(vert_list, fill=(0, 0, 255, 100))

            xmax_p = np.argmax(annotations[k], axis=0)
            x_text = annotations[k][xmax_p[0]][0]
            y_text = annotations[k][xmax_p[0]][1]

            d.text((x_text, y_text), k, fill=(0, 0, 0, 128))

    d.text((int(0.01 * image.size[0]), int(0.01 * image.size[1])),
           "Display Level: {0}".format(str(_level)), fill=(0, 0, 0, 128))

    out = Image.alpha_composite(image, overlay)
    return tiff_image.get_thumbnail(tiff_image.level_dimensions[_level]), out, annot_mask
