# import libraries
import json
import codecs
import requests
import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import os

def convert_vott_csv_to_yolo(
    vott_df,
    labeldict=dict(zip(["Cat_Face"], [0,])),
    path="",
    target_name="data_train.txt",
    abs_path=False,
):

    # Encode labels according to labeldict if code's don't exist
    if not "code" in vott_df.columns:
        vott_df["code"] = vott_df["label"].apply(lambda x: labeldict[x])
    # Round float to ints
    for col in vott_df[["xmin", "ymin", "xmax", "ymax"]]:
        vott_df[col] = (vott_df[col]).apply(lambda x: round(x))

    # Create Yolo Text file
    last_image = ""
    txt_file = ""

    for index, row in vott_df.iterrows():
        if not last_image == row["image"]:
            if abs_path:
                txt_file += "\n" + row["image_path"] + " "
            else:
                txt_file += "\n" + os.path.join(path, row["image"]) + " "
            txt_file += ",".join(
                [
                    str(x)
                    for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
                ]
            )
        else:
            txt_file += " "
            txt_file += ",".join(
                [
                    str(x)
                    for x in (row[["xmin", "ymin", "xmax", "ymax", "code"]].tolist())
                ]
            )
        last_image = row["image"]
    file = open(target_name, "w")
    file.write(txt_file[1:])
    file.close()
    return True

# get links and stuff from json
jsonData = []
JSONPATH = "C:\\Python27\\YOLO\\Data\\Source_Images\\Training_Images_Kaggle\\datasets_36341_54972_face_detection.json"
CSVPATH = "C:\\Python27\\YOLO\\Data\\Source_Images\\Training_Images_Kaggle\\json-csv-export\\Annotation-export.csv"
YOLO_filename = "C:\\Python27\\YOLO\\Data\\Source_Images\\Training_Images_Kaggle\\json-csv-export\\data_train.txt"
classes_filename = "C:\\Python27\\YOLO\\Data\\Model_Weights\\data_classes.txt"
with codecs.open(JSONPATH, 'rU', 'utf-8') as js:
    for line in js:
        jsonData.append(json.loads(line))

print(f"{len(jsonData)} image found!")

print("Sample row:")
jsonData[0]

# load images from url and save into images
IMAGEPATH = "C:\\Python27\\YOLO\\Data\\Source_Images\\Training_Images_Kaggle\\json-csv-export"
images = []
for data in tqdm(jsonData):
    response = requests.get(data['content'])
    img = np.asarray(Image.open(BytesIO(response.content)))
    images.append([img, data["annotation"]])

import cv2
import time
import csv

count = 1
totalfaces = 0
start = time.time()
with open(CSVPATH, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image","xmin","ymin","xmax","ymax","label"])
    for image in images:
        img = image[0]
        metadata = image[1]
        for data in metadata:
            height = data['imageHeight']
            width = data['imageWidth']
            points = data['points']
            if 'Face' in data['label']:
                filename = 'face_image_{}.jpg'.format(count)
                x1 = round(width*points[0]['x'])
                y1 = round(height*points[0]['y'])
                x2 = round(width*points[1]['x'])
                y2 = round(height*points[1]['y'])
                label = data['label']
                writer.writerow([filename, x1, y1, x2, y2, label])
                #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                totalfaces += 1
            
        cv2.imwrite('{}\\face_image_{}.jpg'.format(IMAGEPATH, count), img)
        #cv2.imwrite('/kaggle/output/face-detection-images/face_image_{}.jpg'.format(count),img)
        count += 1
    
end = time.time()

print("Total test images with faces : {}".format(len(images)))
print("Sucessfully tested {} images".format(count-1))
print("Execution time in seconds {}".format(end-start))
print("Total Faces Detected {}".format(totalfaces))

multi_df = pd.read_csv(CSVPATH)
labels = multi_df["label"].unique()
labeldict = dict(zip(labels, range(len(labels))))
multi_df.drop_duplicates(subset=None, keep="first", inplace=True)
train_path = IMAGEPATH
convert_vott_csv_to_yolo(
    multi_df, labeldict, path=train_path, target_name=YOLO_filename
)

# Make classes file
file = open(classes_filename, "w")

# Sort Dict by Values
SortedLabelDict = sorted(labeldict.items(), key=lambda x: x[1])
for elem in SortedLabelDict:
    file.write(elem[0] + "\n")
file.close()
