# Facial-Keypoints-Detection
**Tools and Technology used:** Python, PyTorch, Matplotlib, openCV

Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. Here they are, numbered, and you can see that specific ranges of points match different portions of the face.

![Image added](images/images/landmarks.jpg)

In this project we are defining and training a convolutional neural network to perform facial keypoint detection, and use computer vision techniques to transform images for training and testing.

## How to run the code:
    1. To train and test the model:
       python main.py 
    2. To use the trained model on any random image:
       python fkp.py
    3. To use filters:
       python filters.py

## Results
1. facial key point detection before training (PINK = detected keypoints, GREEN = ground truth):

![Image added](before_train/res0.png)  ![Image added](before_train/res1.png)  ![Image added](before_train/res2.png)

2. facial key point detection after training:

![Image added](saved_images/res0.png)   ![Image added](saved_images/res1.png)   ![Image added](saved_images/res2.png)

3. Facial keypoint detection on a random image:

<img src="images/images/obamas.jpg"  width="350" height="230">

<img src="saved_images/1.jpg"  width="350" height="350">      <img src="saved_images/2.jpg"  width="350" height="350">

4. Applying filters on an image

<img src="saved_images/fkp_detected.png"  width="200" height="200">      <img src="saved_images/final_filtered.png"  width="200" height="200">
