import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
import skimage.io as io
from PIL import Image


class PetaccSplitDataset(Dataset):
    """ Pettacc3 patch dataset."""

    def __init__(self, csv_file, root_dir, transform="Default"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if transform == "Default":
            self.transform = transforms.Compose([
                transforms.Resize(299),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor()
                , transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                       )
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.image_annotations.iloc[idx, 0])

        image = Image.fromarray(io.imread(img_name)).convert("HSV")
        if self.transform:
            image = self.transform(image)
            # image = image.convert("HSV")
        annotation = self.image_annotations.iloc[idx, 1]
        if annotation == "TU":
            annotation = [1]
        else:
            annotation = [0]

        sample = {'image': image, 'annotation': torch.Tensor(annotation)}

        return sample
