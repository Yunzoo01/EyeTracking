# Eye Tracking

## 🌻Description
This program focused on the question, "How can eye-tracking technology be implemented using only a webcam without any additional hardware?" It was judged that using a webcam, which is easily accessible in everyday life, would increase convenience, but the accuracy is lower than using special devices. Therefore, a deep learning model was used to recognize the pupils, and eye-tracking technology was implemented by considering facial movement directions and gaze calibration.

- Use a deep learning model to recognize the movement of the pupils to implement eye-tracking technology.
- Calibrate the gaze (calibration) to implement eye-tracking technology.
- Consider the direction of facial movement to implement eye-tracking technology.
- Research the correlation between the movement of the pupils and the movement on the screen to improve the accuracy of eye-tracking technology.
- Implement a heat map to visualize the data showing where the user was focusing.

## 🌻Getting Started
- Development Environment: Windows 10 Pro
- Development Tools: PyCharm / CUDA 11.7.0
- Technology Stack: Python / TensorFlow / Keras / OpenCV
- Version : Python 10
 
## 🌻Flow Chart
### ▪️ Overall Code Structure
![슬라이드3](https://github.com/user-attachments/assets/73175e6a-084d-4e66-8445-e094921e6ff5)
### ▪️ AI Model Algorithm
![슬라이드2](https://github.com/user-attachments/assets/e2ba163c-47e6-44b3-9244-f870e87264bd)
### ▪️ Eye Tracking Algorithm
![슬라이드1](https://github.com/user-attachments/assets/413e58d8-d1e4-47e9-8413-727427884557)


## 🌻Service Usage Instructions

### ▪️ Calibration
#### First of all, align your face with the designated frame of the camera and press the 's' key.
<img width="350" alt="스크린샷 2024-07-22 오후 3 01 51" src="https://github.com/user-attachments/assets/b36311ca-55ab-43c1-8bfa-484de289407f">

#### When the red dot appears initially, follow the red dot with your eyes only, without moving your head. Next, when the blue dot appears, follow the blue dot with both your head and eyes.
<p align="center">
 <img width="400" alt="스크린샷 2024-07-22 오후 3 01 34" src="https://github.com/user-attachments/assets/1fe53117-e96e-4334-a168-421b0b974529">
 <img width="400" alt="스크린샷 2024-07-22 오후 3 01 42" src="https://github.com/user-attachments/assets/9f40e2df-40b9-49c2-aea3-d559d92f3e0f">
</p>

### ▪️ Gaze tracking
#### Measure the gaze while looking at the designated photos and videos.
<p align="center">
 <img width="400" alt="스크린샷 2024-07-22 오후 3 15 29" src="https://github.com/user-attachments/assets/13bd05fc-e516-4a91-a138-1188952d5a85">
 <img width="400" alt="스크린샷 2024-07-22 오후 3 14 29" src="https://github.com/user-attachments/assets/97ff1961-8d08-4306-bc12-cec90906d02a">
</p>

### ▪️ Result
#### Check the results. The results of the video can be checked through video.py.
<p align="center">
        <img width="400" src="https://github.com/user-attachments/assets/369ca34c-da95-4efb-a800-cbbc5eda57c5" alt="Heatmap">
        <img width="400" src="https://github.com/user-attachments/assets/d118616f-b8ef-426d-b3e1-0833746c3481" alt="Graph">
</p>
<img width="800" alt="스크린샷 2024-07-22 오후 3 16 35" src="https://github.com/user-attachments/assets/eba505af-ea1c-4702-8cfc-edbc7f1ee439">





