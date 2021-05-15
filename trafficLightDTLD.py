# # Traffic Light Classifier
# track_id
import json
import logging

import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, os.path
import math
import numpy as np
import pandas as pd

import yaml

with open('../DTLD_Labels/Bochum_all_jav.yml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
imgs = []
valid_images = [".tiff"]
folder = '../scratch/fs2/DTLD_final/Bochum/Bochum1/2015-04-21_17-09-21/2'

p = 0
for path, subdirs, files in os.walk(folder):
    p += 1
    for f in files:
        if os.path.isfile(os.path.join(path, f)):
            # Load image from file path, do debayering and shift
            img = cv2.imread(os.path.join(path, f), cv2.IMREAD_UNCHANGED)
            # img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR) # 27 out of 215
            img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2RGB)  ####################26 out of 215
            # Images are saved in 12 bit raw -> shift 4 bits
            img = np.right_shift(img, 4)
            img = img.astype(np.uint8)
            imgs.append(img)
        else:
            logging.exception(
                "Image {} not found".format("../scratch/fs2/DTLD_final/Bochum/Bochum1/2015-04-21_17-09-21/2"))

print('Number of pictures: ', len(imgs))

# boxing
tl_image_list = []
tl_label_list = []
k = 0
IMAGE_LIST = []
for j in imgs:
    output = j.copy()
    for i in data[k].get("objects"):
        tl = j[i.get("y"):i.get("y") + i.get("height"), i.get("x"):i.get("x") + i.get("width")]

        cv2.rectangle(output, (i.get("x"), i.get("y")), (i.get("x") + i.get("width"), i.get("y") + i.get("height")),
                      (0, 0, 255), 2)  # doboz
        label = str(i.get("class_id"))
        if label[4] == '0':
            labell = 'off'
        elif label[4] == '1':
            labell = 'red'
        elif label[4] == '2':
            labell = 'yellow'
        elif label[4] == '3':
            labell = 'red-yellow'
        else:
            labell = 'green'
        tl_label_list.append(labell)

        # print(one_hot_encode_dtld(label[4]))

        cv2.putText(output, str(i.get("class_id")), (i.get("x"), i.get("y") - 5),  # lámpaállapot
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        IMAGE_LIST.append([tl, labell])
    k += 1

print('Number of traffic_light: ', len(IMAGE_LIST))


def to_polar_coords(xpix, ypix):
    # Calculate distance to each pixel
    dist = np.sqrt(xpix ** 2 + ypix ** 2)
    # Calculate angle using arctangent function
    angles = np.arctan2(ypix, xpix)
    return dist, angles


# Use a sobel image filter and return the magnitude and angles vectors
# for the image
# reference : https://www.programcreek.com/python/example/89325/cv2.Sobel
def sobel_filter(image):
    # sobel filter x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    # absolute the gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # get magnitude array & angle array
    mag, ang = to_polar_coords(abs_sobelx, abs_sobely)

    # scale values to 8 bit
    scale_factor = np.max(mag) / 255
    mag = (mag / scale_factor).astype(np.uint8)

    return mag, ang


def getBoundingBox(img_thres):
    # ref : https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(img_thres > 0))
    rect = cv2.minAreaRect(coords)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    angle = rect[-1]
    # handle the angle to get correct output
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    return box, angle


def rectanglemask(img, box):
    # transpose as box is flipped
    mask_img = cv2.transpose(np.zeros_like(img))
    cv2.drawContours(mask_img, [box], 0, (255, 255, 255), -1)
    mask_img = np.uint8(mask_img)
    ret, mask = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)
    # transpose back
    mask = cv2.transpose(mask)
    return mask


# rotate the image by a specified angle.
# ref: https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
def rotate(image, angle):
    if len(image.shape) > 2:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


# Some of the traffic light images are very skewed. This function attempts to
# center the traffic lights for easier feature extraction.
# This function filters and image through a sobel image filter extracting magnitude vectors from
# each pixel. These vectors are then filtered and an average image rotation angle is computed
# and returned.
# The magnitude thresold, and angle threshold values were randomly choosen though trail and error.
def getTrafficLightAngle(image, mag_thes=(80, 255), ang_thres=(-np.pi / 6, np.pi / 6)):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mag, ang = sobel_filter(gray)
    ### print(average(mag, mag_thes[0], mag_thes[1]))

    # threshold between angle values, and magnitude values
    ang_threshold = np.zeros_like(ang)
    above_ang = (ang > ang_thres[0]) & (ang <= ang_thres[1]) & (mag > mag_thes[0]) & (mag <= mag_thes[1])
    ang_threshold[above_ang] = 1

    # mask the thresheld image with the angle image
    masked_ang = np.copy(ang)
    masked_ang[ang_threshold == 0] = 0

    h, w = ang_threshold.shape
    # If the thresholding only revelas a few pixels disregard
    # and set angle to zero
    numOfEdges = np.count_nonzero(masked_ang > 0)
    if numOfEdges > 80:
        box, angle = getBoundingBox(ang_threshold)
    else:
        angle = 0
        box = np.array([[0, w],
                        [0, 0],
                        [h, 0],
                        [h, w]], np.int32)

    return box, angle, masked_ang


# CODE TEST
# --------------
# red, _, index = getImage(color='green', Random=True)
test = IMAGE_LIST[1][0]
standard_im = np.copy(test)
box, angle, sobel = getTrafficLightAngle(standard_im)
# plt.imshow(standard_im)
# plt.waitforbuttonpress(0.1)
# plt.imshow(sobel)
# plt.waitforbuttonpress(0.1)

# create rectangluar mask from edge detection
mask = rectanglemask(sobel, box)
mask = cv2.resize(mask, (test.shape[1], test.shape[0]))
mask_img = np.copy(test)
# mask image
mask_img = cv2.bitwise_and(test, mask_img, mask=mask)

# rotate image by skew
cnt = rotate(mask, angle)
# get bounding box
x, y, w, h = cv2.boundingRect(cnt)
# rotate masked image & crop
rotated = rotate(mask_img, angle)
cropped = rotated[y:y + h, x:x + w]


# f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
# ax1.set_title('image')
# ax1.imshow(test)
# ax2.set_title('mask')
# ax2.imshow(mask_img)
# ax3.set_title('cropped')
# ax3.imshow(cropped)
# plt.show()


# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    box, angle, sobel = getTrafficLightAngle(image)

    # create rectangluar mask from edge detection
    mask = rectanglemask(sobel, box)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_img = np.copy(image)
    # mask image
    mask_img = cv2.bitwise_and(image, mask_img, mask=mask)

    # rotate image by skew
    cnt = rotate(mask, angle)
    # get bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    # rotate masked image & crop
    rotated = rotate(mask_img, angle)
    cropped = rotated[y:y + h, x:x + w]

    # resize to 32 X 32
    standard_im = cv2.resize(cropped, (32, 32))

    return standard_im


test = IMAGE_LIST[1][0]
# ax1.set_title('test')
# ax1.imshow(test)
# plt.show()

# original red green and blue image
test = standardize_input(test)


# ax1.set_title('test')
# ax1.imshow(test)
# plt.show()


def one_hot_encode(label):
    one_hot_encoded = [0, 0, 0, 0, 0]
    # check whether color is red green of yellow
    acceptable_colors = tuple(['off', 'red', 'yellow', 'red-yellow', 'green'])
    if (not label in acceptable_colors):
        raise ValueError('label: {} is not an acceptable color'.format(label))
    if label == 'off':
        one_hot_encoded[0] = 1
    elif label == 'red':
        one_hot_encoded[1] = 1
    elif label == 'yellow':
        one_hot_encoded[2] = 1
    elif label == 'red-yellow':
        one_hot_encoded[3] = 1
    else:
        one_hot_encoded[4] = 1

    return one_hot_encoded


def re_one_hot_encode(label):
    one_hot_encoded = 'off'
    # check whether color is red green of yellow
    acceptable_colors = tuple([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    if (not label in acceptable_colors):
        raise ValueError('label: {} is not an acceptable color'.format(label))
    if label == [1, 0, 0, 0, 0]:
        one_hot_encoded = 'off'
    elif label == [0, 1, 0, 0, 0]:
        one_hot_encoded = 'red'
    elif label == [0, 0, 1, 0, 0]:
        one_hot_encoded = 'yellow'
    elif label == [0, 0, 0, 1, 0]:
        one_hot_encoded = 'red-yellow'
    else:
        one_hot_encoded = 'green'

    return one_hot_encoded


def standardize(image_list):
    # Empty image data array
    standard_list = []
    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))
    return standard_list


# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

total_num_of_images = len(STANDARDIZED_LIST)
image_index = random.randint(0, total_num_of_images - 1)

# f2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# ax1.set_title(STANDARDIZED_LIST[image_index][1])
# ax1.imshow(STANDARDIZED_LIST[image_index][0])
# ax2.set_title(IMAGE_LIST[image_index][1])
# ax2.imshow(IMAGE_LIST[image_index][0])

# Convert and image to HSV colorspace
# Visualize the individual color channels
image_num = random.randint(0, total_num_of_images - 1)
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# HSV channels
h = hsv[:, :, 0]
s = hsv[:, :, 1]
v = hsv[:, :, 2]

# Plot the original image and the three channels
# f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
# ax1.set_title('Standardized image')
# ax1.imshow(test_im)
# ax2.set_title('H channel')
# ax2.imshow(h, cmap='gray')
# ax3.set_title('S channel')
# ax3.imshow(s, cmap='gray')
# ax4.set_title('V channel')
# ax4.imshow(v, cmap='gray')

# rgb channels
r = test_im[:, :, 0]
g = test_im[:, :, 1]
b = test_im[:, :, 2]


# Plot the original image and the three channels
# f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
# ax1.set_title('Standardized image')
# ax1.imshow(test_im)
# ax2.set_title('R channel')
# ax2.imshow(r, cmap='gray')
# ax3.set_title('G channel')
# ax3.imshow(g, cmap='gray')
# ax4.set_title('B channel')
# ax4.imshow(b, cmap='gray')


### Helper functions
#### List multiplication

def matrix_scalar_mul(matrix, scalar):
    new = []
    for i in range(len(matrix)):
        new.append(matrix[i] * scalar)
    return new


def matrix_multiplication(matrixA, matrixB):
    product = []

    if len(matrixA) != len(matrixB):
        raise ValueError('list must be the same size, A:', len(matrixA), 'B:', len(matrixB))

    for i in range(len(matrixA)):
        product.append(matrixA[i] * matrixB[i])

    return product


# return list index which has a maximum value
def max_idx(yvals, ranges):
    mx = 0
    j = 0
    for i in ranges:
        if yvals[i] > mx:
            mx = yvals[i]
            j = i
    return j


# return the top x indicies from a list
def max_idx_rank(yvals):
    indicies = set(range(len(yvals)))
    # creat a list to append max bins
    max_list = []
    # create set to perform set operations
    max_set = set()
    intersect = indicies - max_set

    # rank first 8 bins
    for i in range(0, 8):
        # append next maximum value
        max_list.append(max_idx(yvals, intersect))
        # add value to set list
        max_set.add(max_list[-1])
        # remove bin from bin list
        intersect = indicies - max_set

    return max_list


# function to determine if the distribution is bimodal or normal
def is_bimodal(max_list, values):
    difference = []
    for i in range(len(max_list) - 1):
        for j in range(max_list[i], max_list[i + 1]):
            if values[j] == 0:
                return True

    return False


def color_isolate(image, channel):
    if channel == "hsv":
        # Convert absimage to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Create color channels
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        return h, s, v
    else:
        rgb = np.copy(image)
        # Create color channels
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        return r, g, b


def yaxis_hists(rgb_image, channel):
    # seperate image out into different channels of color space
    c1, c2, c3 = color_isolate(rgb_image, 'hsv')
    # Sum components over all coloumns for each row (axis = 1)
    hist_sum = []
    c1_sum = np.sum(c1[:, :], axis=1)
    c2_sum = np.sum(c2[:, :], axis=1)
    c3_sum = np.sum(c3[:, :], axis=1)

    # get baselines
    base1 = np.median(c1_sum)
    base2 = np.median(c2_sum)
    base3 = np.median(c3_sum)

    # split histrogram around the median
    c1_norm = matrix_scalar_mul((c1_sum - base1).tolist(), -1)
    c2_norm = (c2_sum - base2).tolist()
    c3_norm = (c3_sum - base3).tolist()

    # get rid of negative values
    # np.nan
    c1_norm = [0 if x < 0 else x for x in c1_norm]
    c2_norm = [0 if x < 0 else x for x in c2_norm]
    c3_norm = [0 if x < 0 else x for x in c3_norm]

    # package as 2D list
    hist_vals = []
    hist_vals.append(c1_norm)
    hist_vals.append(c2_norm)
    hist_vals.append(c3_norm)

    # get bins
    bin_edges = range(rgb_image.shape[0])

    return bin_edges, hist_vals


def plotHist(bins, values, thickness, channel, yLbl, yLim=False):
    if channel == 'hsv':
        names = ('hue', 'saturation', 'value')
    else:
        names = ('red', 'green', 'blue')

    plt.rcdefaults()
    f1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

    # first channel
    hist1 = ax1.barh(bins, values[0], align="center", height=thickness)
    # get mean and bimodal bool
    max_list = max_idx_rank(values[0])
    mean = max_list[0]
    bimodal = is_bimodal(max_list, values[0])

    ax1.invert_yaxis()
    ax1.set_ylabel(yLbl)
    ax1.set_xlabel("amount")
    ax1.set_title("%s" % names[0])
    ax1.set_title("%s :mu = %i, bi = %r" % (names[0], mean, bimodal))
    if yLim:
        ax1.set_xlim(0, 256)

    # second channel
    hist2 = ax2.barh(bins, values[2], align="center", height=thickness)
    # get mean and bimodal bool
    max_list2 = max_idx_rank(values[2])
    mean2 = max_list2[0]
    bimodal2 = is_bimodal(max_list2, values[2])

    ax2.invert_yaxis()
    ax2.set_ylabel(yLbl)
    ax2.set_xlabel("amount")
    ax2.set_title("%s :mu = %i, bi = %r" % (names[2], mean2, bimodal2))
    if yLim:
        ax2.set_xlim(0, 256)

    # third channel
    sat_val = matrix_multiplication(values[2], values[0])
    hist3 = ax3.barh(bins, sat_val, align="center", height=thickness)
    # get mean and bimodal bool
    max_list3 = max_idx_rank(sat_val)
    mean3 = max_list3[0]
    bimodal3 = is_bimodal(max_list3, sat_val)

    ax3.invert_yaxis()
    ax3.set_ylabel(yLbl)
    ax3.set_xlabel("amount")
    ax3.set_title("%s :mu = %i, bi = %r" % ('hue+val', mean3, bimodal3))
    if yLim:
        ax3.set_xlim(0, 256)

    plt.show()


def feature_value(rgb_image, plot=False):
    # calculate HSVspace over the height on the traffic light
    bins, values = yaxis_hists(rgb_image, 'hsv')

    # for testing purposes
    if plot == True:
        plotHist(bins, values, 0.8, 'hsv', 'y-dist')

    return values[2]


feature = feature_value(test_im, True)
# plt.imshow(test_im)


test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)


# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)
def get_predicted_images(test_images):
    # Track misclassified images by placing them into a list
    predicted_labels = []
    obs_prob = []
    obs_probability = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:
        # plt.imshow(image[0])
        # plt.show()
        # Get true data
        im = image[0]
        true_label = image[1]
        assert (len(true_label) == 5), "The true_label is not the expected length (5)."

        red = 0
        i = 0
        while i != 11:
            t = feature_value(im)
            red = red + t[i]
            i += 1

        yellow = 0
        i = 11
        while i != 20:
            t = feature_value(im)
            yellow = yellow + t[i]
            i += 1

        green = 0
        i = 20
        while i != 31:
            t = feature_value(im)
            green = green + t[i]
            i += 1

        u = math.fabs(red - yellow)
        o = math.fabs(yellow - green)
        p = math.fabs(green - red)

        summ = red + yellow + green
        # print('Össz intenzitás: ', summ)
        prob_off = (1 - (u / summ)) * (1 - (o / summ)) * (1 - (p / summ))
        prob_red = red / summ
        prob_yellow = yellow / summ
        prob_redyellow = ((red + yellow) / summ) * (1 - (u / (red + yellow)))
        prob_green = green / summ

        summa = prob_red + prob_redyellow + prob_off + prob_green + prob_yellow
        # print('redyel:', prob_redyel)
        # obs_prob = [summ/5, summ/5, summ/5, summ/5, summ/5]
        obs_prob = [prob_off / summa, prob_red / summa, prob_yellow / summa, prob_redyellow / summa, prob_green / summa]
        obs_probability.append(
            [prob_off / summa, prob_red / summa, prob_yellow / summa, prob_redyellow / summa, prob_green / summa])
        # print('Valószínűség:', obs_prob)

        # if u < 500 and o < 500 and p < 500:
        if max(obs_prob) == prob_off / summa:
            predicted_label = [1, 0, 0, 0, 0]  # off
        # elif red > 3500 and yellow > 3000:
        elif max(obs_prob) == prob_redyellow / summa:
            predicted_label = [0, 0, 0, 1, 0]  # red-yellow

        # elif max(red, yellow, green) == red:
        elif max(obs_prob) == prob_red / summa:
            predicted_label = [0, 1, 0, 0, 0]  # red

        # elif max(red, yellow, green) == yellow:
        elif max(obs_prob) == prob_yellow / summa:
            predicted_label = [0, 0, 1, 0, 0]  # yellow

        else:
            predicted_label = [0, 0, 0, 0, 1]  # green

        # print('Jósolt:', predicted_label)
        predicted_labels.append((predicted_label, true_label))
        # print(obs_prob)

    return predicted_labels, obs_probability


def viterbi(pi, a, obs_prob):
    nStates = np.shape(a)[0]
    # T = np.shape(obs_prob[1])[0]
    T = len(obs_prob)
    # T = np.shape(obs_prob)[:0]

    # init blank path
    path = np.zeros(T)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))

    # init delta and phi
    delta[:, 0] = pi * obs_prob[0, :]  # b[:, obs[0]]
    # delta[:, 0] = pi * b[:, obs_prob[:,0]]
    phi[:, 0] = 0

    # print('\nStart Walk Forward\n')
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t - 1] * a[:, s]) * obs_prob[t, s]  # * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t - 1] * a[:, s])
            # print('s={s} and t={t}: phi[{s}, {t}] = {phi} ---{delta}'.format(s=s, t=t, phi=phi[s, t], delta=delta[s,t]))

    # find optimal path
    # print('-' * 50)
    # print('Start Backtrace\n')
    path[T - 1] = np.argmax(delta[:, T - 1])
    # p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T - 2, -1, -1):
        path[t] = phi[int(path[t + 1]), [t + 1]]
        # p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        # print('path[{}] = {}'.format(t, path[t]))

    return path, delta, phi


# Find all misclassified images in a given test set
PREDICTED = get_predicted_images(STANDARDIZED_LIST)[0]
PROBS = get_predicted_images(STANDARDIZED_LIST)[1]

k = 0
l = 0
confusion_matrix = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]

traffic_light_names = []
traffic_obs_prob = []
w = 0
total = 0
index = []
for j in imgs:
    output = j.copy()
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    for i in data[k].get("objects"):
        cv2.rectangle(output, (i.get("x"), i.get("y")), (i.get("x") + i.get("width"), i.get("y") + i.get("height")),
                      (0, 0, 255), 2)  # doboz
        label = str(i.get("class_id"))

        if label[4] == '0':
            label = 'off'
        elif label[4] == '1':
            label = 'red'
        elif label[4] == '2':
            label = 'yellow'
        elif label[4] == '3':
            label = 'red-yellow'
        else:
            label = 'green'

        if PREDICTED[l][0] == [1, 0, 0, 0, 0]:
            predicted_label = 'off'
        elif PREDICTED[l][0] == [0, 1, 0, 0, 0]:
            predicted_label = 'red'
        elif PREDICTED[l][0] == [0, 0, 1, 0, 0]:
            predicted_label = 'yellow'
        elif PREDICTED[l][0] == [0, 0, 0, 1, 0]:
            predicted_label = 'red-yellow'
        else:
            predicted_label = 'green'

        if PREDICTED[l][1] == [1, 0, 0, 0, 0]:
            true_label = 'off'
        elif PREDICTED[l][1] == [0, 1, 0, 0, 0]:
            true_label = 'red'
        elif PREDICTED[l][1] == [0, 0, 1, 0, 0]:
            true_label = 'yellow'
        elif PREDICTED[l][1] == [0, 0, 0, 1, 0]:
            true_label = 'red-yellow'
        else:
            true_label = 'green'

        '''cv2.putText(output, label, (i.get("x"), i.get("y") - 30),  # lámpaállapot
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)'''
        if PREDICTED[l][0] != PREDICTED[l][1]:
            cv2.putText(output, true_label, (i.get("x"), i.get("y") - 5),  # lámpaállapot
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(output, predicted_label, (i.get("x"), i.get("y") - 15),  # jósolt
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            w += 1
        else:
            cv2.putText(output, true_label, (i.get("x"), i.get("y") - 5),  # lámpaállapot
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(output, predicted_label, (i.get("x"), i.get("y") - 15),  # lámpaállapot
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        m = 0
        while PREDICTED[l][0][m] != 1:
            m += 1
        n = 0
        while PREDICTED[l][1][n] != 1:
            n += 1
        confusion_matrix[m][n] = confusion_matrix[m][n] + 1

        volt = False
        o = 0
        for z in traffic_light_names:
            if i.get("track_id") == z:
                volt = True
            # traffic_obs_prob.insert(o, PROBS[l])
            # break
            # o += 1

        if volt == False:
            traffic_light_names.append(i.get("track_id"))

        l += 1
        total += 1

    # cv2.imwrite("../estimated/image"+str(k)+".tiff", output)
    # cv2.waitKey(0)
    k += 1

print('off, red, yellow, red-yellow, green')
for i in confusion_matrix:
    print(*i)

o = 0
fail = 0
all_true = []
all_pred = []
all_mpred = []
fin_true = []
f_true = []
f_pred = []
f_mpred = []
f_obs = []
indexx = []
for m in traffic_light_names:
    k = 0
    l = 0
    traffic_obs_prob_m = []
    traffic_obs_prob_pred = []
    traffic_obs_prob_true = []
    index = []
    for j in imgs:
        for i in data[k].get("objects"):
            if i.get("track_id") == m:
                traffic_obs_prob_m.append(PROBS[l])
                traffic_obs_prob_pred.append(PREDICTED[l][0])
                traffic_obs_prob_true.append(PREDICTED[l][1])
                index.append(i.get("unique_id"))
            l += 1
        k += 1
    print(m)
    print(traffic_obs_prob_m)
    # print(traffic_obs_prob_pred)
    traffic_obs_prob.append(traffic_obs_prob_m)

    # markov
    hidden_states = ['off', 'red', 'yellow', 'red-yellow', 'green']
    pi = [0.2, 0.2, 0.2, 0.2, 0.2]
    state_space = pd.Series(pi, index=hidden_states, name='states')
    # print(state_space)
    # print('\n', state_space.sum())

    # hidden_state transition matrix (valamilyen állapotból melyikbe mehet át, kb. egyenletes eloszlás mivel nem tudunk semmit)
    a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
    a_df.loc[hidden_states[0]] = [0.2, 0.2, 0.2, 0.2, 0.2]
    a_df.loc[hidden_states[1]] = [0.3, 0.35, 0, 0.35, 0]
    a_df.loc[hidden_states[2]] = [0.3, 0.35, 0.35, 0, 0]
    a_df.loc[hidden_states[3]] = [0.3, 0, 0, 0.35, 0.35]
    a_df.loc[hidden_states[4]] = [0.3, 0, 0.35, 0, 0.35]

    # print(a_df)

    a = a_df.values
    # print('\n', a, a.shape, '\n')
    # print(a_df.sum(axis=1))

    observable_states = ['off', 'red', 'yellow', 'red-yellow', 'green']

    b_df = pd.DataFrame(columns=observable_states, index=hidden_states)

    # observations are encoded numerically
    obs_map = {'off': 0, 'red': 1, 'yellow': 2, 'red-yellow': 3, 'green': 4}
    # obs = np.array([0,0,1,1,1,3,3,3,4,4,2,4,4,4,4,1,2,2,2,4,1])
    '''obs_prob = [[0.3,0.5,0.2,0.2,0.1,0.1,0.1,0.1,0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                [0.2,0.2,0.4,0.5,0.6,0.2,0.2,0.1,0.1,0.1,0.2,0.2,0.1,0.1,0.1,0.3,0.2,0.2,0.1,0.2,0.4],
                [0.2,0.1,0.2,0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.3,0.1,0.1,0.1,0.1,0.2,0.4,0.5,0.6,0.2,0.2],
                [0.1,0.1,0.1,0.1,0.1,0.3,0.4,0.5,0.2,0.2,0.2,0.2,0.2,0.1,0.1,0.2,0.2,0.1,0.1,0.2,0.2],
                [0.2,0.1,0.1,0.1,0.1,0.2,0.1,0.1,0.3,0.4,0.2,0.4,0.5,0.6,0.6,0.2,0.1,0.1,0.1,0.3,0.1]]'''
    obs_prob = np.array(traffic_obs_prob_m)
    obs_seq = np.array(traffic_obs_prob_pred)
    seq_true = traffic_obs_prob_true

    inv_obs_map = dict((v, k) for k, v in obs_map.items())

    # path, delta, phi = viterbi(pi, a, b, obs, obs_prob)
    path, delta, phi = viterbi(pi, a, obs_prob)
    # print('\nsingle best state path: \n', path)
    # print('delta:\n', delta)
    # print('phi:\n', phi)

    state_map = {0: [1, 0, 0, 0, 0], 1: [0, 1, 0, 0, 0], 2: [0, 0, 1, 0, 0], 3: [0, 0, 0, 1, 0], 4: [0, 0, 0, 0, 1]}
    state_path = [state_map[v] for v in path]
    s_mpred = [re_one_hot_encode(v) for v in state_path]
    s_true = [re_one_hot_encode(v) for v in seq_true]
    s_pred = [re_one_hot_encode(v) for v in traffic_obs_prob_pred]
    all_true.append(s_true)
    all_pred.append(s_pred)
    all_mpred.append(state_path)
    print('Jósolt, szekvencia: ', s_pred)
    print('Valós, szekvencia : ', s_true)
    print('M javított szekven: ', s_mpred)

    for x in s_true:
        f_true.append(x)
    for x in s_pred:
        f_pred.append(x)
    for x in s_mpred:
        f_mpred.append(x)
    for x in index:
        indexx.append(x)
    for x in traffic_obs_prob_m:
        f_obs.append(x)

    for i in range(len(seq_true)):
        if seq_true[i] != state_path[i]:
            fail += 1

# (pd.DataFrame()
# .assign(Observation=obs_seq)
# .assign(Best_Path=state_path))

o += 1

k = 0
l = 0
for j in imgs:
    output = j.copy()
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    for i in data[k].get("objects"):
        h = 0
        label = 'nincs'
        for x in indexx:
            if i.get("unique_id") == x:
                cv2.rectangle(output, (i.get("x"), i.get("y")), (i.get("x") + i.get("width"), i.get("y") + i.get("height")),
                              (0, 0, 255), 2)  # doboz
                cv2.rectangle(output, (i.get("x")+i.get("width")+int(50*f_obs[h][0]), i.get("y")),
                              (i.get("x") + i.get("width")+5, i.get("y") + 5),
                              (0, 0, 0), 2)  # off-doboz

                cv2.putText(output, str(round(f_obs[h][0],2)), (i.get("x")+65, i.get("y")),  # off
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)

                cv2.rectangle(output, (i.get("x") + i.get("width") + int(50*f_obs[h][1]), i.get("y")+10),
                              (i.get("x") + i.get("width")+5, i.get("y") + 15),
                              (0, 0, 255), 2)  # red-doboz

                cv2.putText(output, str(round(f_obs[h][1],2)), (i.get("x") + 65, i.get("y")+10),  # red
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)

                cv2.rectangle(output, (i.get("x") + i.get("width") + int(50*f_obs[h][2]), i.get("y") + 20),
                              (i.get("x") + i.get("width")+5, i.get("y") + 25),
                              (0, 255, 255), 2)  # yellow-doboz

                cv2.putText(output, str(round(f_obs[h][2],2)), (i.get("x") + 65, i.get("y") + 20),  # yellow
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)

                cv2.rectangle(output, (i.get("x") + i.get("width") + int(50*f_obs[h][3]), i.get("y") + 30),
                              (i.get("x") + i.get("width")+5, i.get("y") + 35),
                              (0, 140, 255), 2)  # red-yellow-doboz

                cv2.putText(output, str(round(f_obs[h][3],2)), (i.get("x") + 65, i.get("y") + 30),  # red-yellow
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 140, 255), 1)

                cv2.rectangle(output, (i.get("x") + i.get("width") + int(50*f_obs[h][4]), i.get("y") + 40),
                              (i.get("x") + i.get("width")+5, i.get("y") + 45),
                              (0, 255, 0), 2)  # green-doboz

                cv2.putText(output, str(round(f_obs[h][4],2)), (i.get("x") + 65, i.get("y") + 40),  # green
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)


                label = str(i.get("class_id"))
                cv2.putText(output, str('val:'+ f_true[h]), (i.get("x"), i.get("y") - 5),  # valós
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if f_true[h] != f_pred[h]:
                    cv2.putText(output, str('becs:'+ f_pred[h]), (i.get("x"), i.get("y") - 20),  # jósolt rossz
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.putText(output, str('becs:'+f_pred[h]), (i.get("x"), i.get("y") - 20),  # jósolt jó
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if f_true[h] != f_mpred[h]:
                    cv2.putText(output, str('jav:'+f_mpred[h]), (i.get("x"), i.get("y") - 35),  # javított rossz
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.putText(output, str('jav:'+f_mpred[h]), (i.get("x"), i.get("y") - 35),  # javított jó
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(output, i.get('track_id'), (i.get("x"), i.get("y") - 50),  # lámpa id
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            h += 1

        l += 1
    import os

    cv2.imwrite("../estimated_3/image" + str(k) + ".tiff", output)
    cv2.waitKey(0)
    k += 1

#print(index)
#print(f_true)
#print(len(indexx), len(f_true))
print(100 - (w * 100 / total), "%")
print(100 - (fail * 100 / total), "%")
