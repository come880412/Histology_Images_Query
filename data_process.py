import json, os, random, tqdm, shutil
import numpy as np
import re
import cv2
from PIL import Image, ImageOps, ImageFilter

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

def data_augmentation():
    from torchvision import transforms
    from PIL import Image

    img = cv2.imread("../dataset/train/000fa8e6db0e.png")
    cv2.imwrite('./ori.jpg', img)
    
    img = Image.open("../dataset/train/000fa8e6db0e.png").convert("RGB")
    # img = transforms.RandomResizedCrop(224)(img)

    img10 = transforms.RandomHorizontalFlip()(img)
    img10.save('./Horizontal.jpg')

    imgB = transforms.ColorJitter(brightness=0.5)(img)
    imgB.save('./brightness.jpg')
    imgC = transforms.ColorJitter(contrast=0.5)(img)
    imgC.save('./contrast.jpg')
    imgS = transforms.ColorJitter(saturation=0.5)(img)
    imgS.save('./saturation.jpg')
    imgH = transforms.ColorJitter(hue=0.5)(img)
    imgH.save('./hue.jpg')
    imgall = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(img)
    imgall.save('./ColorJitter.jpg')

    img_gaussian = transforms.GaussianBlur((23, 23), (0.1, 2.0))(img)
    img_gaussian.save('./gaussian.jpg')

    img_solor = Solarization(1)(img)
    img_solor.save('./solar.jpg')

    img_gray = transforms.RandomGrayscale(1)(img)
    img_gray.save('./grayscale.jpg')

if __name__ in "__main__":
    data_augmentation() # Visualize the image after data augmentation
