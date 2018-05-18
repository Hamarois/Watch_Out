#!/usr/bin/python
# -*- coding: utf-8 -*-
# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import dlib
import cv2
import serial
import time

# set arduino port
port = '/dev/ttyUSB0'
ard = serial.Serial(port,9600,timeout=5)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# argument parsin
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

ear_blink = 0.3 # eye aspect ratio for blink
ear_below_frames = 48 # 눈을 아래로 깔고있는(감고있는) frame 수

# frame counter 초기화
cnt = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("start predictor")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("video stream thread is starting")
vs = VideoStream(src=1).start() # set usb camera
time.sleep(1.0)

# loop over frames from the video stream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	faces = detector(gray, 0)

	# loop over the face detections
	for face in faces:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, face)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)

		# visualize eyes contours on frame
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < ear_blink:
			cnt += 1

			# 눈을 일정 frame 이상 감고있으면
			if cnt >= ear_below_frames:

				cv2.putText(frame, "Sleep!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

				# Serial Communication with arduino
				light = "1"
				ard.write(light.encode())

				# Serial read section
				msg = ard.readline()
				print ("Message from arduino: ", msg)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			cnt = 0

			# Serial Communication with arduino
			light = "0"
			ard.write(light.encode())

			# Serial read section
			msg = ard.readline()
			print ("Message from arduino: ", msg)

		# draw the computed eye aspect ratio on the frame
		# cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

