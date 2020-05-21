import cv2
import torch
from torchvision.transforms import transforms
from microscopyio.slide_image import NDPISlideImage
import matplotlib
from PIL import Image


def classify_image(model, device, image_path, mask_path, annotation_path, name, patch_size=1024):
    image = NDPISlideImage(image_path=image_path, tissue_mask_path=mask_path
                           # , tumor_annotation_file=annotation_path
                           )
    patches = image._compute_patch_cover(
        probe_level=6,
        patch_size=patch_size,
        patch_shift=patch_size,
        min_coverage=0.95
    )

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]
    )

    def get_prediction_for_patch(patch_pos):
        patch = Image.fromarray(image.load_patch(patch_pos, (patch_size, patch_size), 0)).convert("HSV")
        nn_ready_img = transform(patch)[None, ...].to(device)
        res = torch.round(model(nn_ready_img)).item()
        if res == 1.0:
            res = 'TU'
        else:
            res = 'NO'
        return res

    labeled_patches = list(map(lambda patch: [patch, get_prediction_for_patch(patch)], patches))

    p_img = image.get_patch_visualization(6, labeled_patches, patch_size, show=False)
    cv2.imwrite(name, p_img)
