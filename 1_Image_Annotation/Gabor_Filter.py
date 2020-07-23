from matplotlib import pyplot as plt
import cv2
import csv
import numpy as np
from random import randrange, uniform
import pandas as pd

IMAGEPATH = "C:\\Python27\\YOLOandGabor\\Data\\Source_Images\\Training_Images_Kaggle\\Resize_Images"
CSVPATH = "C:\\Python27\\YOLOandGabor\\Data\\Source_Images\\Training_Images_Kaggle\\Resize_Images\\Annotation-export-resize.csv"
target_csv = 'C:\\Python27\\YOLOandGabor\\Data\\Source_Images\\Training_Images_Kaggle\\Filtered_Image\\Annotation-export-filtered.csv'
target = 'C:\\Python27\\YOLOandGabor\\Data\\Source_Images\\Training_Images_Kaggle\\Filtered_Image'

num = 7
kernels = []
for i in range(num * num):
    kernels.append(cv2.getGaborKernel((100, 100), 4, randrange(0,180, 45), randrange(10, 20), uniform(.2, .9), 0, ktype=cv2.CV_32F))

# Credits: https://stackoverflow.com/a/42579291/8791891
def convolution2d(image, kernel, bias):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel) + bias
    return new_image

fig, ax = plt.subplots(num, num, sharex=True, sharey=True, figsize=[16, 16])


file_in = pd.read_csv(CSVPATH)
with open(target_csv, 'w', newline='') as file_out:
    writer = csv.writer(file_out)
    # save the new bounding box resized images
    writer.writerow(["image","xmin","ymin","xmax","ymax","label"])
    before = ''
    for index, data in file_in.iterrows():
        image = data['image']
        xmin = data['xmin']
        ymin = data['ymin']
        xmax = data['xmax']
        ymax = data['ymax']
        label = data['label']
        k = 0
        #img = cv2.imread(f"{IMAGEPATH}\\{image}", cv2.IMREAD_GRAYSCALE)
        for i in range(num * num):
            filtered_name = f"filtered-{k}-{image}"
            #if before != data['image']:
            #    plt.imsave(f"{target}\\{filtered_name}", convolution2d(img, cv2.resize(kernels[k], (7, 7), cv2.INTER_AREA), 1), cmap='gray')
            writer.writerow([filtered_name, xmin, ymin, xmax, ymax, label])
            k += 1
        before = image
