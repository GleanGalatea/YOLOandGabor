import cv2
import os
import pandas as pd
import csv

newWidth = 128
newHeight = 128
dim = (newWidth, newHeight)

IMAGEPATH = "C:\\Python27\\YOLOandGabor\\Data\\Source_Images\\Training_Images_Kaggle\\json-csv-export"
CSVPATH = "C:\\Python27\\YOLOandGabor\\Data\\Source_Images\\Training_Images_Kaggle\\json-csv-export\\Annotation-export.csv"
target_csv = 'C:\\Python27\\YOLOandGabor\\Data\\Source_Images\\Training_Images_Kaggle\\Resize_Images\\Annotation-export-resize.csv'
target = 'C:\\Python27\\YOLOandGabor\\Data\\Source_Images\\Training_Images_Kaggle\\Resize_Images'

# Credits : https://www.kaggle.com/kevinpatel04/convert-json-annotation-to-csv
file_in = pd.read_csv(CSVPATH)
with open(target_csv, 'w', newline='') as file_out:
    writer = csv.writer(file_out)
    # save the new bounding box resized images
    writer.writerow(["image","xmin","ymin","xmax","ymax","label"])
    for index, data in file_in.iterrows():
        image = data['image']
        img = cv2.imread(f"{IMAGEPATH}\\{image}", cv2.IMREAD_UNCHANGED)
        height = img.shape[0]
        width = img.shape[1]
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        xmin = data['xmin']
        ymin = data['ymin']
        xmax = data['xmax']
        ymax = data['ymax']
        if 'Face' in data['label']:
            xminAfter = round((newWidth*xmin)/width)
            yminAfter = round((newHeight*ymin)/height)
            xmaxAfter = round((newWidth*xmax)/width)
            ymaxAfter = round((newHeight*ymax)/height)
            label = data['label']
            resized_name = f"resized-{image}"
            writer.writerow([resized_name, xminAfter, yminAfter, xmaxAfter, ymaxAfter, label])
        cv2.imwrite(f'{target}\\{resized_name}', resized)
        