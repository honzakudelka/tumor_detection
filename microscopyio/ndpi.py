import os.path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from collections import namedtuple

import numpy as np
from openslide import OpenSlide, OpenSlideError

NDPIMeta = namedtuple("NDPIMeta", "Offset Levels LevelDimensions DownsamplingFactor MPP")


def read_file(path):
    """
    Reads an ndpi image from drive

    :param path: Full path to the NDPI file
    :return: a ndpi_image object and its metadata
    """

    ndpi_image = None
    try:
        ndpi_image = OpenSlide(path)
    except OpenSlideError as oe:
        print ("[OpenSlideException] " + str(oe))
    except IOError as e:
        print ("Failed to read data, \n [IO-Exception] {0}".format(str(e)))
    finally:
        if ndpi_image is None:
            raise RuntimeError("[Failed to load ndpi image] " + path)
    x_offset = int(ndpi_image.properties['hamamatsu.XOffsetFromSlideCentre'])
    y_offset = int(ndpi_image.properties['hamamatsu.YOffsetFromSlideCentre'])
    x_mpp = float(ndpi_image.properties['openslide.mpp-x'])
    y_mpp = float(ndpi_image.properties['openslide.mpp-y'])

    metadata = NDPIMeta((x_offset, y_offset), ndpi_image.level_count,
                        ndpi_image.level_dimensions,
                        ndpi_image.level_downsamples,
                        (x_mpp, y_mpp))

    return ndpi_image, metadata


def read_annotation_file(annonfile, img_meta, level):
    xml_file = ET.parse(annonfile)
    xml_root = xml_file.getroot()

    annotations = {}
    count = 0

    for annotation in list(xml_root):
        # name = annotation.find('title').text
        p = annotation.find('annotation')
        if p is None:
            continue
        if p.find('closed').text != "1":
            continue

        plist = p.find('pointlist')
        if p is None:
            continue

        xy_coords = []
        for points in list(plist):
            x = int(points.find('x').text)
            y = int(points.find('y').text)

            x = (x - img_meta.Offset[0]) / (1000 * img_meta.MPP[0])
            y = (y - img_meta.Offset[1]) / (1000 * img_meta.MPP[1])

            x_px = (x + img_meta.LevelDimensions[0][0] / 2) / img_meta.DownsamplingFactor[level]
            y_px = (y + img_meta.LevelDimensions[0][1] / 2) / img_meta.DownsamplingFactor[level]

            xy_coords.append([x_px, y_px])

        if len(xy_coords) < 5:
            # too short
            continue

        # check the last point to match the first one
        if (xy_coords[0][0] != xy_coords[-1][0]) or (xy_coords[0][1] != xy_coords[-1][1]):
            xy_coords.append(xy_coords[0])

        xy_coords = np.array(xy_coords, dtype=np.int32)

        ann_name = "Annotation_{0}".format(str(count))
        count += 1

        annotations.update({ann_name: xy_coords})

    return annotations


# def extract_annotated_region(file, annonfile=None, level=0):
#
#     ndpi_image, metadata = read_file(file)
#
#     if annonfile is None and os.path.exists(file+".ndpa"):
#         annonfile = file+".ndpa"
#
#     try:
#         annotations = read_annotation_file(file+".ndpa", metadata, _level)
#     except IOError as e:
#         print "Failed to read data, \n [IO-Exception] {0}".format(str(e))
#


def show_image(file, level=0, ann_ext_list=(".ndpa",)):
    """

    :param file: Path to NDPI file
    :param level: Zoom level to be extracted (default: highest resolution, 0)
    :param ann_ext_list: List of extensions to probe when searching for annotation file
    :return:
    """

    ndpi_image, metadata = read_file(file)
    _level = min(level, metadata.Levels - 1)

    annotations = None
    for ann_ext in ann_ext_list:
        if os.path.exists(file + ann_ext):
            annotations = read_annotation_file(file + ann_ext, metadata, _level)
            break

    image = ndpi_image.get_thumbnail(ndpi_image.level_dimensions[_level]).convert('RGBA')
    if annotations is None:
        return image

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))

    d = ImageDraw.Draw(overlay)

    for k in annotations:
        vert_list = []
        for i, point in enumerate(annotations[k]):

            start = point
            vert_list.append((start[0], start[1]))

            if i < len(annotations[k]) - 1:
                end = annotations[k][i + 1]
            else:
                continue

            d.line((start[0], start[1], end[0], end[1]), fill=(255, 0, 0, 128), width=8)

        # enable if the polygon should be filled
        # d.polygon(vert_list, fill=(0, 0, 255, 100))

        xmax_p = np.argmax(annotations[k], axis=0)
        x_text = annotations[k][xmax_p[0]][0]
        y_text = annotations[k][xmax_p[0]][1]

        d.text((x_text, y_text), k, fill=(0, 0, 0, 128))

    out = Image.alpha_composite(image, overlay)
    return out
