from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import pandas as pd
import PIL
import os.path as path
import numpy as np

class ImageDataset(Dataset):

    def __init__(self, usedImageIds=None, dataAugmentation=None, trainImagesDir=None, hasTrainLabels=True):
        self.dataAugmentation = dataAugmentation
        self.csvFileName = "file_to_labels_table.csv"
        self.image_id_and_labels = pd.read_csv(self.csvFileName, index_col=False)
        file_path = path.abspath("image_dataset.py")
        self.trainImagesDir = trainImagesDir if trainImagesDir is not None else path.join(path.dirname(file_path), "train/images")
        self.usedImageIds = usedImageIds if usedImageIds is not None else range(1,len(self.image_id_and_labels)+1)
        self.hasTrainLabels = hasTrainLabels

    def __len__(self):
        return len(self.usedImageIds)

    def __getitem__(self, idx):
        id = int(self.usedImageIds[idx])
        tensorLabels = None
        stringLabels = None
        if self.hasTrainLabels:
            labels = self.image_id_and_labels[self.image_id_and_labels.columns[1:]].iloc[id-1].values

            stringLabels = [self.image_id_and_labels.columns[i+1] for i in range(len(labels)) if labels[i] == 1]
            tensorLabels = torch.tensor(labels)

        imageFileName = "im{}.jpg".format(id)

        imagePath = path.join(self.trainImagesDir, imageFileName)
        image = PIL.Image.open(imagePath)
        image = image.convert("RGB")
        image = image.resize((224,224))

        if self.dataAugmentation:
            image = PIL.Image.fromarray(self.dataAugmentation.augment_image(np.array(image)))

        imageTensor = transforms.functional.to_tensor(image)
        imageTensor = transforms.functional.normalize(imageTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return {"imageTensor": imageTensor, "imagePil": image, "labelsTensor": tensorLabels, "labelsString": stringLabels}
    
    def set_images_path(self, abs_path=None):
        self.trainImagesDir = abs_path if abs_path is not None else path.join(path.dirname(path.abspath("image_dataset.py")), "train/images")
