"""

"""

import argparse
import csv
from microscopyio.slide_image import NDPISlideImage
from PIL import Image
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract patches from microscopy (histopat.) images")

    parser.add_argument("-f", "--folder", type=str, help='Image folder path', default=None, required=True)
    parser.add_argument("-o", "--output_folder", type=str, help='Output folder path', default=None, required=True)

    opt_group = parser.add_argument_group("Optional arguments")

    opt_group.add_argument("-c", "--csv", type=str, help='CSV file path', default="default", required=False)
    opt_group.add_argument("-a", "--augmentation", type=bool, help='Augment dataset', default=True, required=False)
    # opt_group.add_argument("-a", "--anotations", type=str, help='Annotation folder path', default="default", required=False)
    # opt_group.add_argument("-m", "--masks", type=str, help='Mask folder path', default="default", required=False)
    # opt_group.add_argument("-i", "--images", type=str, help='Image folder path', default="default", required=False)

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

    csv_path = cmdline_args.csv
    if cmdline_args.csv == "default":
        csv_path = (cmdline_args.folder + "/overview.csv")

    p_shift = cmdline_args.patch_stride
    output_folder = cmdline_args.output_folder
    if p_shift < 0:
        p_shift = cmdline_args.patch_size
    if os.path.isfile(f"{output_folder}\\validation.csv"):
        os.remove(f"{output_folder}\\validation.csv")
    if os.path.isfile(f"{output_folder}\\training.csv"):
        os.remove(f"{output_folder}\\training.csv")
    if os.path.isfile(f"{output_folder}\\testing.csv"):
        os.remove(f"{output_folder}\\testing.csv")
    validate_csv = open(f"{output_folder}\\validation.csv", "w+")
    validate_csv.write("Image,Annotation\n")
    train_csv = open(f"{output_folder}\\training.csv", "w+")
    train_csv.write("Image,Annotation\n")
    test_csv = open(f"{output_folder}\\testing.csv", "w+")
    test_csv.write("Image,Annotation\n")

    augmentation = cmdline_args.augmentation
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif line_count < 84:
                split_row = row[0].split(';')
                if split_row[1] == 'None' and cmdline_args.skip:
                    line_count += 1
                    continue

                image_path = (cmdline_args.folder + "\\" + split_row[0] + ".ndpi").replace("/", "\\").replace("\\\\",
                                                                                                              "\\")
                annotation_path = (cmdline_args.folder + "\\" + split_row[1]).replace("/", "\\").replace("\\\\", "\\")
                mask_path = (cmdline_args.folder + "\\" + split_row[2]).replace("/", "\\").replace("\\\\", "\\")
                cat_muc = split_row[3]
                cat_ser = split_row[4]
                cat_cro = split_row[5]
                si = NDPISlideImage(image_path=image_path,
                                    tissue_mask_path=mask_path, tumor_annotation_file=annotation_path)
                p = si.get_annotated_patches(extract_level=cmdline_args.extract_level,
                                             min_coverage_extraction=cmdline_args.min_coverage,
                                             min_tumor_coverage=0.9, p_size=cmdline_args.patch_size, p_shift=p_shift)
                name = split_row[0].split("/")[1].split("-")[0]
                patch_count = 0
                all_count = 1
                tucount = 0
                nocount = 0
                bocount = 0
                for patch in p:

                    patch_array = si.load_patch(patch[0], (cmdline_args.patch_size, cmdline_args.patch_size))
                    image = Image.fromarray(patch_array)
                    if augmentation:
                        images = [image, image.transpose(Image.FLIP_LEFT_RIGHT), image.transpose(Image.FLIP_TOP_BOTTOM),
                                  image.transpose(Image.ROTATE_90), image.transpose(Image.ROTATE_180),
                                  image.transpose(Image.ROTATE_270)]
                    else:
                        images = [image]
                    if patch[1] == 'TU':
                        if all_count % 5 == 0:
                            for image in images:
                                image_name = f"{name}_{tucount}_TU.jpg"
                                image.save(f"{output_folder}\\test\\{image_name}")
                                test_csv.write(f"{image_name},TU\n")
                                tucount += 1
                            all_count = 1
                        else:
                            for image in images:
                                image_name = f"{name}_{tucount}_TU.jpg"
                                image.save(f"{output_folder}\\train\\{image_name}")
                                train_csv.write(f"{image_name},TU\n")
                                tucount += 1
                            all_count += 1

                    elif patch[1] == 'NO':
                        if all_count % 5 == 0:
                            for image in images:
                                image_name = f"{name}_{nocount}_NO.jpg"
                                image.save(f"{output_folder}\\test\\{image_name}")
                                test_csv.write(f"{image_name},NO\n")
                                nocount += 1
                            all_count = 1
                        else:
                            for image in images:
                                image_name = f"{name}_{nocount}_NO.jpg"
                                image.save(f"{output_folder}\\train\\{image_name}")
                                train_csv.write(f"{image_name},NO\n")
                                nocount += 1
                            all_count += 1
                    elif patch[1] == 'BO':
                        if all_count % 5 == 0:
                            for image in images:
                                image_name = f"{name}_{nocount}_BO.jpg"
                                image.save(f"{output_folder}\\test\\{image_name}")
                                test_csv.write(f"{image_name},NO\n")
                                bocount += 1
                            all_count = 1
                        else:
                            for image in images:
                                image_name = f"{name}_{nocount}_BO.jpg"
                                image.save(f"{output_folder}\\train\\{image_name}")
                                train_csv.write(f"{image_name},BO\n")
                                bocount += 1
                            all_count += 1

                    patch_count += 1
                    if patch_count % 50 == 0:
                        print(f"{patch_count} patches extracted from image no {line_count}/150")
                line_count += 1
            else:
                split_row = row[0].split(';')

                if split_row[1] == 'None' and cmdline_args.skip:
                    line_count += 1
                    continue

                image_path = (cmdline_args.folder + "\\" + split_row[0] + ".ndpi").replace("/", "\\").replace("\\\\",
                                                                                                              "\\")
                annotation_path = (cmdline_args.folder + "\\" + split_row[1]).replace("/", "\\").replace("\\\\", "\\")
                mask_path = (cmdline_args.folder + "\\" + split_row[2]).replace("/", "\\").replace("\\\\", "\\")
                cat_muc = split_row[3]
                cat_ser = split_row[4]
                cat_cro = split_row[5]
                si = NDPISlideImage(image_path=image_path,
                                    tissue_mask_path=mask_path, tumor_annotation_file=annotation_path)
                p = si.get_annotated_patches(extract_level=cmdline_args.extract_level,
                                             min_coverage_extraction=cmdline_args.min_coverage,
                                             min_tumor_coverage=0.9, p_size=cmdline_args.patch_size, p_shift=p_shift)

                name = split_row[0].split("/")[1].split("-")[0]
                patch_count = 0
                all_count = 1
                tucount = 0
                nocount = 0
                for patch in p:

                    patch_array = si.load_patch(patch[0], (cmdline_args.patch_size, cmdline_args.patch_size))
                    image = Image.fromarray(patch_array)
                    if augmentation:
                        images = [image, image.transpose(Image.FLIP_LEFT_RIGHT), image.transpose(Image.FLIP_TOP_BOTTOM),
                                  image.transpose(Image.ROTATE_90), image.transpose(Image.ROTATE_180),
                                  image.transpose(Image.ROTATE_270)]
                    else:
                        images = [image]
                    if patch[1] == 'TU':
                        for image in images:
                            image_name = f"{name}_{tucount}_TU.jpg"
                            image.save(f"{output_folder}\\validate\\{image_name}")
                            validate_csv.write(f"{image_name},TU\n")
                            tucount += 1

                    elif patch[1] == 'NO':

                        for image in images:
                            image_name = f"{name}_{nocount}_NO.jpg"
                            image.save(f"{output_folder}\\validate\\{image_name}")
                            validate_csv.write(f"{image_name},NO\n")
                            nocount += 1

                    elif patch[1] == 'BO':

                        for image in images:
                            image_name = f"{name}_{nocount}_BO.jpg"
                            image.save(f"{output_folder}\\validate\\{image_name}")
                            validate_csv.write(f"{image_name},NO\n")
                            bocount += 1

                    patch_count += 1
                    if patch_count % 50 == 0:
                        print(f"{patch_count} patches extracted from image no {line_count}/150")
                line_count += 1
        print(f'Processed {line_count} images.')
