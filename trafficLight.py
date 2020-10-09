from msilib import Directory

import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images
import yaml
import os, os.path
import json
import test_functions

###############################BDD#########################################################
# json fájl betöltése
with open('../bdd100k/labels/bdd100k_labels_images_val.json', 'r') as f:
    data = json.loads(f.read())

#for i in data[0].get("labels"):
#    if i.get("category") == "traffic light":
#        print(i.get("box2d")["x1"])

#képek betöltése
'''imgs = []
valid_images = [".jpg"]
for path, subdirs, files in os.walk("../bdd100k_images/bdd100k/images/100k/3/"):
    for f in files:
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue

        imgs.append(cv2.imread(os.path.join(path, f)))
#cv2.imshow("image",imgs[0])
#cv2.waitKey(0)

output = imgs[0].copy()
for i in data[0].get("labels"):
    if i.get("category") == "traffic light":
        tl = imgs[1][int(i.get("box2d")["y1"]):int(i.get("box2d")["y2"]),int(i.get("box2d")["x1"]):int(i.get("box2d")["x2"])]
        #cv2.imshow(i.get("track_id"), tl)

        cv2.rectangle(output, (int(i.get("box2d")["x1"]), int(i.get("box2d")["y1"])), (int(i.get("box2d")["x2"]), int(i.get("box2d")["y2"])), (0, 0, 255), 2) #doboz

        cv2.putText(output, i.get("attributes")["trafficLightColor"], (int(i.get("box2d")["x1"]), int(i.get("box2d")["y1"]-5)),     #lámpaállapot
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

cv2.imshow("TrafficLights", output)
cv2.waitKey(0)'''



###############################DTLD########################################################
#yaml fájl betöltése


with open('../DTLD_Labels/Bochum_all.yml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    #print(yaml.dump(data[1]))


#képek lista betöltése
imgs = []
valid_images = [".tiff"]
for path, subdirs, files in os.walk("../scratch/fs2/DTLD_final/Bochum/Bochum1/2015-04-21_17-09-21/2"):
    for f in files:
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue

        imgs.append(cv2.imread(os.path.join(path, f)))

output = imgs[1].copy()


def standardize_input(image):
    standard_im = np.copy(image)
    return cv2.resize(standard_im, (32,32))

def one_hot_encode_dtld(label):
    colors = {"0": 0,
              "1": 1,
              "2": 2,
              "3": 3,
              "4": 4}
    one_hot_encoded = [0] * len(colors)
    one_hot_encoded[colors[label]] = 1

    return one_hot_encoded

def one_hot_encode_bdd(label):
    colors = {"red": 0,
              "yellow": 1,
              "green": 2,
              "none": 3}
    one_hot_encoded = [0] * len(colors)
    one_hot_encoded[colors[label]] = 1

    return one_hot_encoded

# Test for one_hot_encode function
tests = test_functions.Tests()
tests.test_one_hot(one_hot_encode_dtld)


def standardize(image_list, label_list):
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs

    i = 0
    for image in image_list:
        #image = item[0]
        label = label_list[i]
        i = i+1
        # Standardize the image
        standardized_im = standardize_input(image)
        # One-hot encode the label
        one_hot_label = one_hot_encode_dtld(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list


#dobozolás
tl_image_list = []
tl_label_list = []
for i in data[1].get("objects"):
    tl = imgs[1][i.get("y"):i.get("y") + i.get("height"),
         i.get("x"):i.get("x") + i.get("width")]
    standardize_input(tl)
    tl_image_list.append(tl)
    #cv2.imshow(i.get("track_id"), standardize_input(tl))

    cv2.rectangle(output, (i.get("x"), i.get("y")), (i.get("x") + i.get("width"), i.get("y") + i.get("height")), (0, 0, 255), 2) #doboz
    label = str(i.get("class_id"))
    tl_label_list.append(label[4])
    #print(one_hot_encode_dtld(label[4]))

    cv2.putText(output, str(i.get("class_id")), (i.get("x"), i.get("y")-5),     #lámpaállapot
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

cv2.imshow("TrafficLights", output)
cv2.waitKey(0)

# Standardize all training images
STANDARDIZED_LIST = standardize(tl_image_list, tl_label_list)
## Display a standardized image and its label
n = 1
#selected_label = STANDARDIZED_LIST[n][1]
#while selected_label != "yellow":
#    n += 1
#    selected_label = STANDARDIZED_LIST[n][1]

selected_image = STANDARDIZED_LIST[n][0]
plt.imshow(selected_image)
plt.waitforbuttonpress(0.1)
selected_label = STANDARDIZED_LIST[n][1]
print(selected_label)

# Convert and image to HSV colorspace
# Visualize the individual color channels
image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
plt.waitforbuttonpress(0.1)
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')
