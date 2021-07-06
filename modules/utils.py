from xml.etree import ElementTree
from xml.dom import minidom

import numpy as np
from cv2 import cv2

import os
import random
import shutil
import time

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem)
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml()

def img_size(img):
    img = cv2.imread(img)
    return img.shape

def split_dataset(path, percent=10):
    """Randomly split dataset to default 90% trainig data and 10% validation data.
    
    - path - head folder for dataset
    - percent - percent of training data to split

    """
    imagesDir = os.path.join(path, 'train/images')
    annotationsDir = os.path.join(path, 'train/annotations')
    outputImagesDir = os.path.join(path, 'validation/images')
    outputAnnotationsDir = os.path.join(path, 'validation/annotations')

    os.makedirs(outputImagesDir, exist_ok=True)
    os.makedirs(outputAnnotationsDir, exist_ok=True)
    # os.makedirs(os.path.join(path, 'cache'), exist_ok=True)

    try:
        for f in os.listdir(os.path.join(path, 'cache')):
            os.remove(os.path.join(path, 'cache', f))
    except:
        pass

    validation_files = os.listdir(outputImagesDir)
    if len(validation_files) > 0:
        for f in validation_files:
            root, ext = os.path.splitext(f)
            
            g = str(root + '.xml')
            shutil.move(os.path.join(outputImagesDir, f), os.path.join(imagesDir, f))
            shutil.move(os.path.join(outputAnnotationsDir, g), os.path.join(annotationsDir, g))

    files = os.listdir(imagesDir)
    k = int(percent / 100 * len(files))
    index = random.sample(files, k)
    # print(index, k)
    for f in index:
        root, ext = os.path.splitext(f)
        
        g = str(root + '.xml')
        shutil.move(os.path.join(imagesDir, f), os.path.join(outputImagesDir, f))
        shutil.move(os.path.join(annotationsDir, g), os.path.join(outputAnnotationsDir, g))
        
    print("|------------------------------------------------------------|")
    print("| Training and validation data were reshuffled and splitted. |")
    print("|------------------------------------------------------------|")
    time.sleep(10)

# if __name__ == '__main__':
#     split_dataset('C:/Users/Aaros/OneDrive/PycharmProjects/Projekt/carsv3')