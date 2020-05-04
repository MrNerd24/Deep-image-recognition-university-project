from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import pandas as pd
import PIL
import os.path as path
import numpy as np

class ImageDataset(Dataset):

    def __init__(self, usedImageIds=None, dataAugmentation=None):
        self.dataAugmentation = dataAugmentation
        self.csvFileName = "image_id_and_labels.csv"
        self.image_id_and_labels = pd.read_csv(self.csvFileName, index_col=False)
        file_path = path.abspath("image_dataset.py")
        self.trainImagesDir = path.join(path.dirname(file_path), "train/images")
        self.usedImageIds = usedImageIds if usedImageIds else range(1,len(self.image_id_and_labels)+1)

    def __len__(self):
        return len(self.usedImageIds)

    def __getitem__(self, idx):
        id = self.usedImageIds[idx]
        row = self.image_id_and_labels.iloc[id-1].values

        labels = row[1:]
        stringLabels = [self.image_id_and_labels.columns[i+1] for i in range(len(labels)) if labels[i] == 1]
        tensorLabels = torch.tensor(labels)
        imageId = row[0]

        imagePath = path.join(self.trainImagesDir, "im{}.jpg".format(str(imageId)))
        image = PIL.Image.open(imagePath)
        image = image.convert("RGB")
        image = image.resize((224,224))

        if self.dataAugmentation:
            image = PIL.Image.fromarray(self.dataAugmentation.augment_image(np.array(image)))

        imageTensor = transforms.functional.to_tensor(image)
        imageTensor = transforms.functional.normalize(imageTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return {"imageTensor": imageTensor, "imagePil": image, "labelsTensor": tensorLabels, "labelsString": stringLabels}
