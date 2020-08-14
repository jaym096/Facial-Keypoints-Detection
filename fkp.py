import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from models import Net

# load in color image for face detection
image = cv2.imread('images/obamas.jpg')

# switch red and blue color channels
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot the image
fig = plt.figure(figsize=(9,9))
plt.savefig("saved_images/normal_img.jpg")

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# run the detector
# the output here is an array of detections; the corners of each detection box
faces = face_cascade.detectMultiScale(image, 1.2, 2)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()

# loop over the detected faces, mark the image where each face is found
for (x,y,w,h) in faces:
    # draw a rectangle around each detected face
    # you may also need to change the width of the rectangle drawn depending on image resolution
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)

fig = plt.figure(figsize=(9,9))
plt.savefig("saved_images/face_detected_img.jpg")

net = Net()
net.load_state_dict(torch.load('keypoints_model_final.pt'))
## print out your net and prepare it for testing
net.eval()

image_copy = np.copy(image)

# loop over the detected faces from your haar cascade
count = 0
for (x, y, w, h) in faces:
    count += 1

    # Select the region of interest that is the face in the image
    roi = image_copy[y:y + h, x:x + w]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    roi_gray_copy = np.copy(roi_gray)

    roi_gray = roi_gray / 255

    roi_gray = cv2.resize(roi_gray, (224, 224))
    # print(roi_gray.shape)

    roi_reshp = np.reshape(roi_gray, (1, 224, 224))
    roi_gray = torch.FloatTensor(roi_reshp)
    # print(roi_gray.shape)

    # if the following line of the code is not included, a Runtime Error is encountered.
    roi_gray = roi_gray.unsqueeze(0)

    key_pts = net(roi_gray)
    key_pts = key_pts.view(68, 2)

    # Undo tranformation of keypoints
    key_pts = key_pts.data.numpy()
    key_pts = key_pts * 80.0 + 70

    plt.imshow(roi_gray_copy, cmap='gray')
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
    plt.savefig("saved_images/final_FKP"+str(count)+".jpg")
    plt.clf()