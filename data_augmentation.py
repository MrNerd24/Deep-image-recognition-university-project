import numpy as np
from skimage.transform import AffineTransform, warp, radon
from skimage.util import img_as_ubyte, random_noise

class DataAugmentation():

    def __init__(self, augmentation_probability = 0.8, padding="wrap", move_probability = 0.3, move_variance = 1, rotation_probability=0.3, rotation_variance=0.05, scale_probability=0.3, scale_variance=0.1, shear_probability=0.3, shear_variance=0.1, flip_probability=0.5, noise_probability=0.3, noise_variance=0.01):
        self.augmentation_probability = augmentation_probability
        self.padding = padding

        self.move_probability = move_probability
        self.move_variance = move_variance

        self.rotation_probability = rotation_probability
        self.rotation_variance = rotation_variance

        self.scale_probability = scale_probability
        self.scale_variance = scale_variance

        self.shear_probability = shear_probability
        self.shear_variance = shear_variance

        self.flip_probability = flip_probability

        self.noise_probability = noise_probability
        self.noise_variance = noise_variance

    def augment_image(self, image):
        if np.random.binomial(1, self.augmentation_probability) == 0:
            return img_as_ubyte(image)

        if (np.random.binomial(1, self.noise_probability) == 1):
            image = random_noise(image, var=self.noise_variance)

        if (np.random.binomial(1, self.flip_probability) == 1):
            image = image[:, ::-1]
        
        image = self.transform_image(image)

        return img_as_ubyte(image)

    def transform_image(self, image):
        translation = np.random.normal(0, self.move_variance, 2)*np.random.binomial(1, self.move_probability)
        rotation = np.random.normal(0, self.rotation_variance, 1)*np.random.binomial(1, self.rotation_probability)
        scale = np.ones((2)) + np.random.normal(0, self.scale_variance, 2)*np.random.binomial(1, self.scale_probability)
        shear = np.random.normal(0, self.shear_variance, 1)*np.random.binomial(1, self.shear_probability)
        transform = AffineTransform(translation=translation, rotation=rotation, scale=scale, shear=shear)
        return warp(image, transform, mode=self.padding)
