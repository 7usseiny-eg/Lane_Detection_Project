# Lane_Detection_Project

## Introduction
  * A simple digital image processing project for detecting the lanes and cars across the road. This project has two parts, first part is the notebook based code whereas the second part is the python file run with a three possible arguments, one for the image file path and the other for the video file path and last is the debugging mode

## Requirements
 * For this project to take said image/video and work, you have to download a single folder and place it along with your run.py file(if you didn't download it, then you should) in the same folder.
 * The folder will be found inside the repo named camera_cal.
 * You should NOT change any of the names of either camera_cal or the contents inside of it.

## Debugging Mode
  * Debugging mode is used to know the stages of how the output video was released.
  * You can enable the debugging mode on running by using '1' argument or disable it by not inserting or inserting '0' as an argument
  * For the debugging mode, we show 4 different stages as following:
       1. Warped Image
       2. Threshold Image
       3. Out Image
       4. Color Wrap
  
## Guide
 * Open your terminal and change directory to the folder which contains run.py and camera_cal folder together
 * You can manipulate three arguments: --img [img\file\path] or --vid [vid\file\path] or --debug [0/1]
 * Use the following format to run your file: python run.py --vid "file\path" --img "file\path" --debug 0/1
     Examples: 
     1. python run.py --img "C:\Users\Omar\Downloads\Lane_detection\test1.jpg" --debug 0
     2. python run.py --vid "C:\Users\Omar\Downloads\Lane_detection\project_video.mp4" --debug 1
     3. python run.py --img "C:\Users\Omar\Downloads\Lane_detection\test1.jpg"--vid "C:\Users\Omar\Downloads\Lane_detection\project_video.mp4" --debug 1
     4. python run.py #THIS EXAMPLE WILL DO NOTHING
 * In case of having an image and a video, the image will be done first and you have to press Q to exit, same goes for the video if you want to close it press Q
