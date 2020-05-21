import cv2
import torch
from torchvision.transforms import transforms
from microscopyio.slide_image import NDPISlideImage
from classify_image import classify_image
import argparse
import os
import csv

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Classify folder of whole slide images")

    parser.add_argument("-f", "--folder", type=str, help='Image folder path', default=None, required=True)
    parser.add_argument("-o", "--output_folder", type=str, help='Output folder path', default=None, required=True)
    parser.add_argument("-clf", "--classifier_path", type=str, help='Classifier path', default=None, required=True)
    opt_group = parser.add_argument_group("Optional arguments")

    opt_group.add_argument("-c", "--csv", type=str, help='CSV file path', default="default", required=False)

    opt_group.add_argument("--skip", type=bool, help='Skip images without annotation', default=True, required=False)
    opt_group.add_argument("-p", "--patch-size", type=int, help='Size of the patch (on the base level 0)',
                           default=1024, required=False)
    opt_group.add_argument("-s", "--patch-stride", type=int, help='Offset to next patch (on the base level 0)',
                           default=-1, required=False)
    opt_group.add_argument("--min-coverage", type=float, help='Minimal tissue coverage in accepted patches',
                           default=0.95, required=False)
    opt_group.add_argument("-l", "--extract-level", type=int, help='Level for extracting patches',
                           default=6, required=False)

    cmdline_args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(cmdline_args.classifier_path, map_location=torch.device(device))
    output_folder = cmdline_args.output_folder
    patch_size = cmdline_args.patch_size
    csv_path = cmdline_args.csv
    if cmdline_args.csv == "default":
        csv_path = (cmdline_args.folder + "/overview.csv")

    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                split_row = row[0].split(';')
                if split_row[1] != 'None' and cmdline_args.skip:
                    line_count += 1
                    continue
                image_name = f"classified_{line_count}b.jpg"
                image_path = (cmdline_args.folder + "\\" + split_row[0] + ".ndpi").replace("/", "\\").replace("\\\\",
                                                                                                              "\\")
                annotation_path = (cmdline_args.folder + "\\" + split_row[1]).replace("/", "\\").replace("\\\\", "\\")
                mask_path = (cmdline_args.folder + "\\" + split_row[2]).replace("/", "\\").replace("\\\\", "\\")
                classify_image(model, device, image_path, mask_path, annotation_path, f"{output_folder}\\{image_name}",
                               patch_size)

                line_count += 1
