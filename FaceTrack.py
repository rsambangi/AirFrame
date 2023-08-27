# Author: Rushil Sambangi
# Description: Face tracking drone media framing tool. Program allows the drone to frame the subject perfectly in the
# center of a video/image solely by tracking the position of the face. A hands-free approach to drone maneuvering and
# alternative to using a controller. MADE FOR USE WITH RYZE TELLO API.

# Import necessary libraries
import cv2
import numpy as np
from djitellopy import tello
from time import sleep

# Establish drone connection
drone = tello.Tello()
drone.connect()

# Begin drone video stream and initial hover height
drone.streamon()
drone.takeoff()
drone.send_rc_control(0, 0, 20, 0)
sleep(1.5)

# Initialize video frame size and initial PID values for drone speed controls
w, h = 360, 240
fbRange = [6200, 6800]
pid = [0.4, 0.4, 0]
pError = 0


# Face detection function using haarcascade algorithm
def findFace(img):
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")  # Load in xml file
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)  # Detect face bounds

    myFaceListC = []  # Array for center coordinates
    myFaceListArea = []  # Array for area

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw bounding box around face
        cx = x + w // 2  # Calculate x center
        cy = y + h // 2  # Calculate y center
        area = w * h  # Calculate total area
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)  # Check area of face to ensure drone is not too close or far from subject
    if len(myFaceListArea) != 0:  # If subject is detected and area exists, return the frame and coordinates
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

# Drone framing and tracking algorithm
def frameFace(info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0

    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)  # Input PID values for smooth acceleration
    speed = int(np.clip(speed, -100, 100))

    # If area is too large, then move drone back and vice versa
    if fbRange[0] < area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20

    # If no face found, drone hovers
    if x == 0:
        speed = 0
        error = 0

    drone.send_rc_control(0, fb, 0, speed)  # Send signals to drone
    return error

# Continuously follow and frame face within video capture
while True:
    img = drone.get_frame_read().frame  # Get frame from video
    img = cv2.resize(img, (w, h))
    img, info = findFace(img)  # Find face
    pError = frameFace(info, w, pid, pError)  # Follow face and frame it within the video
    cv2.imshow("Output", img)  # Show video stream
    if cv2.waitKey(1) & 0xFF == ord('l'):  # Stop and land when user clicks "l"
        drone.land()
        break
