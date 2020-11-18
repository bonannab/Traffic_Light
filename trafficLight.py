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
import logging


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
              "none": 2}
    one_hot_encoded = [0] * len(colors)
    one_hot_encoded[colors[label]] = 1

    return one_hot_encoded

# Test for one_hot_encode function
tests = test_functions.Tests()
#tests.test_one_hot(one_hot_encode_dtld)


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
        one_hot_label = one_hot_encode_dtld(label)      #dtld
        #one_hot_label = one_hot_encode_bdd(label)       #bdd


        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list


## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
def create_feature(rgb_image):
    ##Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    #plt.imshow(rgb_image)
    #plt.waitforbuttonpress(0.1)

    #rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
   # plt.imshow(rgb)
    #plt.waitforbuttonpress(0.1)

    #hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
   # plt.imshow(hsv)
   # plt.waitforbuttonpress(0.1)

    #hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
   # plt.imshow(hsv)
    #plt.waitforbuttonpress(0.1)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Detect edges in S
    # 3x3 edge detection filters
    kernel = np.array([[-4, -4, -4],
                       [-4, 32, -4],
                       [-4, -4, -4]])
    s_edges = cv2.filter2D(s, -1, kernel)


    # Blur edges.  Need to blur enough so that areas with signification changes in saturation bleed into each other
    blur = np.array([[1 / 9, 1 / 9, 1 / 9],
                     [1 / 9, 1 / 9, 1 / 9],
                     [1 / 9, 1 / 9, 1 / 9]])
    s_blur = cv2.filter2D(s, -1, kernel)

    for i in range(20):
        s_blur = cv2.filter2D(s_blur, -1, blur)

    # Create mask based on blurred edges in s
    s_blur_avg = int(np.sum(s_blur) / (len(s_blur) * len(s_blur[0])))
    s_blur_std = int(np.std(s_blur))
    s_mask = np.greater(s_blur, s_blur_avg + s_blur_std)
    #print('s_mask', str(s_mask))
    #print('blur:' + str(s_blur))
    #print('avg:' + str(s_blur_avg))
    #print(s_blur_std)

    # a
    v_mask = v
    v_mask[s_mask == 0] = [0]

    #print(v_mask)

    ## Create feature vector, the brightness in each third of the image after masking
    v_top = np.sum(v_mask[0:15])
    v_middle = np.sum(v_mask[7:23])
    v_bottom = np.sum(v_mask[15:31])
    v_sum = v_top + v_middle + v_bottom
    feature = [v_top / v_sum, v_middle / v_sum, v_bottom / v_sum]
    #print(v_top)
   # print(v_middle)
   # print(v_bottom)
   # print(v_sum)
    #print('feature'+str(feature))

    # visualizing my pipeline
    # f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
    # ax1.set_title('Standardized image')
    # ax1.imshow(rgb_image)
    # ax2.set_title('S channel')
    # ax2.imshow(s, cmap='gray')
    # ax3.set_title('S edges')
    # ax3.imshow(s_edges, 'gray')
    # ax4.set_title('S mask')
    # ax4.imshow(s_mask, 'gray')
    # ax5.set_title('V mask')
    # ax5.imshow(v_mask, 'gray')

    return feature


# feature 2 is average hue using the same mask as in feature 1
def create_feature2(rgb_image):
    ##Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Detect edges in S
    # 3x3 edge detection filters
    kernel = np.array([[-4, -4, -4],
                       [-4, 32, -4],
                       [-4, -4, -4]])
    s_edges = cv2.filter2D(s, -1, kernel)


    # Blur edges.  Need to blur enough so that areas with signification changes in saturation bleed into each other
    blur = np.array([[1 / 9, 1 / 9, 1 / 9],
                     [1 / 9, 1 / 9, 1 / 9],
                     [1 / 9, 1 / 9, 1 / 9]])
    s_blur = cv2.filter2D(s, -1, kernel)
    for i in range(20):
        s_blur = cv2.filter2D(s_blur, -1, blur)

    # Create mask based on blurred edges in s
    s_blur_avg = int(np.sum(s_blur) / (len(s_blur) * len(s_blur[0])))
    s_blur_std = int(np.std(s_blur))
    s_mask = np.greater(s_blur, s_blur_avg + s_blur_std)

    # apply the mask to h
    h_mask = h
    h_mask[s_mask == 0] = [0]

    feature = np.sum(h_mask / 360) / np.sum(s_mask)

    # visualizing my pipeline
    #f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20,10))
    # ax1.set_title('Standardized image')
    # ax1.imshow(rgb_image)
    # ax2.set_title('S channel')
    # ax2.imshow(s, cmap='gray')
    # ax3.set_title('S edges')
    # ax3.imshow(s_edges, 'gray')
    # ax4.set_title('S mask')
    # ax4.imshow(s_mask, 'gray')
    # ax5.set_title('H mask')
    # ax5.imshow(h_mask, 'gray')

    return feature


# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    feature = np.array(create_feature(rgb_image))
    #print(feature.argmax(axis=0))
    predicted_label = [0, 0, 0, 0, 0]

    #print(create_feature2(rgb_image))
 #   if create_feature2(rgb_image) > 0.38:
    #predicted_label[0] = 1
    #if abs(feature[1] - feature[2]) < 0.055):
     #   predicted_label[3] = 1
    if feature.argmax(axis=0)+1 == 3:
        predicted_label[feature.argmax(axis=0) + 2] = 1
    else:
        predicted_label[feature.argmax(axis=0) + 1] = 1

    return predicted_label


# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(image_list):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in image_list:

        # Get true data
        im = image[0]
        true_label = [0, 0, 0, 0, 0]
        true_label[0] = image[1][0]
        true_label[1] = image[1][1]
        true_label[2] = image[1][2]
        true_label[3] = image[1][3]
        true_label[4] = image[1][4]


        assert (len(true_label) == 5), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert (len(predicted_label) == 5), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels
        if (predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))

    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels

###############################BDD#########################################################
# json fájl betöltése
'''with open('../bdd100k/labels/bdd100k_labels_images_val.json', 'r') as f:
    data = json.loads(f.read())

##for i in data[0].get("labels"):
    #if i.get("category") == "traffic light":
      #  print(i.get("box2d")["x1"])

#képek betöltése
imgs = []
valid_images = [".jpg"]
for path, subdirs, files in os.walk("../bdd100k_images/bdd100k/images/100k/3/"):
    for f in files:
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue

        imgs.append(cv2.imread(os.path.join(path, f)))
#cv2.imshow("image",imgs[0])
#cv2.waitKey(0)'''



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
            if os.path.isfile(os.path.join(path, f)):
                # Load image from file path, do debayering and shift
                img = cv2.imread(os.path.join(path, f),cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)
                # Images are saved in 12 bit raw -> shift 4 bits
                img = np.right_shift(img, 4)
                img = img.astype(np.uint8)
                imgs.append(img)
            else:
                logging.exception(
                    "Image {} not found".format("../scratch/fs2/DTLD_final/Bochum/Bochum1/2015-04-21_17-09-21/2"))

        #imgs.append(cv2.imread(os.path.join(path, f)))
#cv2.imshow("lights", imgs[1])
#cv2.waitKey(0)
#output = imgs[0].copy()

#dobozolás DTLD
tl_image_list = []
tl_label_list = []
k=0
for j in imgs:
    output = j.copy()
    for i in data[k].get("objects"):
        tl = j[i.get("y"):i.get("y") + i.get("height"), i.get("x"):i.get("x") + i.get("width")]
        standardize_input(tl)
        tl_image_list.append(standardize_input(tl))
        #cv2.imshow(i.get("track_id"), standardize_input(tl))

        cv2.rectangle(output, (i.get("x"), i.get("y")), (i.get("x") + i.get("width"), i.get("y") + i.get("height")), (0, 0, 255), 2) #doboz
        label = str(i.get("class_id"))
        tl_label_list.append(label[4])
        #print(one_hot_encode_dtld(label[4]))

        cv2.putText(output, str(i.get("class_id")), (i.get("x"), i.get("y")-5),     #lámpaállapot
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        #tl_label_list.append(str(i.get("class_id"))[5])
        cv2.putText(output, str(estimate_label(standardize_input(tl))), (i.get("x"), i.get("y") - 15),  # lámpaállapot
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
    k += 1

    cv2.imshow("TrafficLights", output)
    cv2.waitKey(0)

#dobozolás BDD
'''tl_image_list = []
tl_label_list = []
#output = imgs[0].copy()
k=0
for j in imgs:
    output = j.copy()
    for i in data[k].get("labels"):
        if i.get("category") == "traffic light":
            tl = j[int(i.get("box2d")["y1"]):int(i.get("box2d")["y2"]),int(i.get("box2d")["x1"]):int(i.get("box2d")["x2"])]
            standardize_input(tl)
            tl_image_list.append(standardize_input(tl))
            #cv2.imshow("traffic light" + str(i), standardize_input(tl))

            cv2.rectangle(output, (int(i.get("box2d")["x1"]), int(i.get("box2d")["y1"])), (int(i.get("box2d")["x2"]), int(i.get("box2d")["y2"])), (0, 0, 255), 2) #doboz
            tl_label_list.append(i.get("attributes")["trafficLightColor"])
            cv2.putText(output, i.get("attributes")["trafficLightColor"], (int(i.get("box2d")["x1"]), int(i.get("box2d")["y1"]-5)),     #lámpaállapot
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
            cv2.putText(output, str(estimate_label(standardize_input(tl))),
                        (int(i.get("box2d")["x1"]), int(i.get("box2d")["y1"] - 15)),  # lámpaállapot
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)

    k = k+1
    cv2.imshow("TrafficLights", output)
    cv2.waitKey(0)'''




# Standardize all training images
STANDARDIZED_LIST = standardize(tl_image_list, tl_label_list)
## Display a standardized image and its label
n = 0
#selected_label = STANDARDIZED_LIST[n][1]
#while selected_label != "yellow":
#    n += 1
#    selected_label = STANDARDIZED_LIST[n][1]
k=0
for i in STANDARDIZED_LIST:
    selected_image = i[0]
    selected_label = i[1]
   # print(selected_label)

    test_im = i[0]
    test_label = i[1]

    hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)
   ##### #print('Label [red, yellow, green, none]: ' + str(test_label))

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

   # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
    #ax1.set_title('Standardized image')
   # ax1.imshow(test_im)
   # ax2.set_title('H channel')
   # ax2.imshow(h, cmap='gray')
   # ax3.set_title('S channel')
   # ax3.imshow(s, cmap='gray')
   # ax4.set_title('V channel')
   # ax4.imshow(v, cmap='gray')

    create_feature(test_im)
    create_feature2(test_im)
    if estimate_label(test_im) != test_label:
        plt.imshow(selected_image)
        plt.waitforbuttonpress(0.1)
       # cv2.imshow('teszt',selected_image)
       # cv2.waitKey(0)
        print('......................................')
        print(estimate_label(test_im))
        print('Label [none, red, yellow, yellow-red, green]: ' + str(test_label))
        print(create_feature(test_im))

# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_LIST)

# Accuracy calculations
total = len(STANDARDIZED_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct / total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) + ' out of ' + str(total))

#plt.show(imgs[5])
#plt.waitforbuttonpress(0.1)
# Visualize misclassified example(s)
## TODO: Display an image in the `MISCLASSIFIED` list
## TODO: Print out its predicted label - to see what the image *was* incorrectly classified as
n = 0
#selected_image = MISCLASSIFIED[n][0]
#print(create_feature2(selected_image))
#plt.imshow(selected_image)
#plt.waitforbuttonpress(0.1)

# selected_label = MISCLASSIFIED[n][1]
# print(selected_label)
