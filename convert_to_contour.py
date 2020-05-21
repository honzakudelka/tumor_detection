import os
import sys

import cv2
import numpy as np

from microscopyio import ndpi

if __name__ == "__main__":

    ndpi_root = '/datagrid/Medical/microscopy/petacc3/batch_2/'

    annfile = sys.argv[1]
    ann_level = int(sys.argv[2])
    bname = os.path.basename(annfile)

    ndpi_file = ndpi_root + bname[11:19] + ".ndpi"

    try:
        ndpi_image, metadata = ndpi.read_file(ndpi_file)
    except RuntimeError as e:
        print(e)
        quit(2)

    img = cv2.imread(annfile, cv2.IMREAD_ANYDEPTH)
    imgray = np.uint8(10 * img)

    im2, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_NONE)

    print("Saving contour...")

    ds_factor = metadata.DownsamplingFactor[ann_level]

    xml_str = '<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n'
    ann_list = "<annotations>\n"

    for ci, cnt in enumerate(contours):
        ann_list += '<ndpviewstate id="{}">\n'.format(ci)
        ann_list += "<title>cnt_{}</title>\n".format(ci)
        ann_list += "<annotation>\n<closed>1</closed>\n"
        plist = "<pointlist>"
        for point in cnt:
            x_coord = np.int32((ds_factor * point[0][0] - metadata.LevelDimensions[0][0] / 2) * 1e3 * metadata.MPP[0]
                               + metadata.Offset[0])
            y_coord = np.int32((ds_factor * point[0][1] - metadata.LevelDimensions[0][1] / 2) * 1e3 * metadata.MPP[1]
                               + metadata.Offset[1])

            point_str = "<point>\n<x>{}</x>\n<y>{}</y>\n</point>\n".format(
                x_coord, y_coord
            )
            plist += point_str

        plist += "</pointlist>\n"
        ann_list += plist + "</annotation>\n"
        ann_list += "</ndpviewstate>"
    ann_list += "</annotations>"

    xml_file = ndpi_file + ".xml"

    fh = open(xml_file, 'w')
    fh.write(xml_str + ann_list)
    fh.close()
