Lane Detection Project
Overview
This project implements a lane detection system using OpenCV and Python. The goal is to identify and highlight the lanes in a video feed, typically from a dashcam, to assist in autonomous driving or driver-assistance systems.

Features
Edge Detection: Utilizes the Canny edge detection algorithm to identify edges in the video frames.
Region of Interest: Focuses on a specific area of the frame to improve the accuracy of lane detection.
Hough Transform: Detects lines in the image and identifies lanes based on the detected lines.
Video Processing: Processes video input frame by frame to detect and highlight lanes in real-time.
Demo

(Replace with an actual GIF or screenshot of your project in action)

Installation
To run this project on your local machine, follow these steps:

Prerequisites
Python 3.x
OpenCV
NumPy
Install Dependencies
Make sure you have Python installed. Then, install the required Python packages:

bash
Copy code
pip install opencv-python numpy
Running the Project
Clone the Repository

bash
Copy code
git clone https://github.com/Mohammed-Faazil-16/Lane-Detection.git
cd lane-detection
Run the Lane Detection Script

You can run the script on a test video provided in the repository or on your own video:

bash
Copy code
python lane_detection.py --input test_video.mp4
The output video with the detected lanes will be saved as output_video.mp4.

How It Works
1. Canny Edge Detection
The script first converts the video frames to grayscale and applies Gaussian blur to reduce noise. It then uses the Canny edge detection algorithm to identify edges in the frame.

2. Region of Interest
A region of interest (ROI) is defined to focus on the road where lanes are typically located. This helps in filtering out irrelevant edges from other parts of the frame.

3. Hough Line Transform
The Hough Line Transform is applied to detect lines in the ROI. The detected lines are then averaged and extended to form the final lane lines.

4. Video Processing
The process is repeated for each frame in the video. The resulting frames are then combined and saved as an output video.

Configuration
You can modify several parameters in the script to fine-tune the lane detection process:

Canny Edge Detection Thresholds: Adjust the lower and upper thresholds for edge detection.
Region of Interest: Change the coordinates of the polygon to focus on different parts of the frame.
Hough Transform Parameters: Adjust the parameters like minLineLength and maxLineGap to refine line detection.
Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or suggestions.

License
This project is licensed under the MIT License - see the LICENSE file for details.
