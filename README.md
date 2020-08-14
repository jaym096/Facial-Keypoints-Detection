# Facial-Keypoints-Detection
used implemented a CNN based facial keypoint detection

**Tools and Technology used:** Python, PyTorch, Matplotlib, openCV

## How to run the code:
    1. To train and test the model:
       python main.py 
    2. To use the trained model on any random image:
       python fkp.py
    3. To use filters:
       python filters.py

## Results
1. facial key point detection before training (PINK = detected keypoints, GREEN = ground truth):

![Image added](before_train/res0.png)   ![Image added](before_train/res1.png)   ![Image added](before_train/res2.png)

2. facial key point detection after training:

![Image added](saved_images/res0.png)   ![Image added](saved_images/res1.png)   ![Image added](saved_images/res2.png)

3. Facial keypoint detection on a random image:

![Image added](images/images/obamas.jpg)   ![Image added](saved_images/1.jpg)   ![Image added](saved_images/2.jpg)
