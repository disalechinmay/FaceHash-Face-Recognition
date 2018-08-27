import cv2
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import math
from PIL import Image
from subprocess import call
import os
import threading
import time
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = keras.models.load_model('kerasFaceHash')

# Basically finds distance between 2 points
# Arguments:
# 	-> tempshape: DLIB's predictor which plots facial landmark points
# 	-> point1 & point2: Points between which distance is to be found out
def getDistance(tempshape, point1, point2):
	point1x = tempshape.part(point1).x
	point1y = tempshape.part(point1).y
	point2x = tempshape.part(point2).x
	point2y = tempshape.part(point2).y
				
	dx = (point2x - point1x) ** 2
	dy = (point2y - point1y) ** 2
	distance = math.sqrt(dx + dy)
	return distance


#Importing Haar cascade and DLIB's facial landmarks detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Precision in % -> Tunes the recognizer according to our need
precision = 0.95

targetPoints = []

# Read map.txt -> Can be tuned!
# map.txt holds a collection of points which will be used to recognize a face
# map.txt holds a list of pairs between which will define a set of lines to be considered by recognizer
lines = [line.rstrip('\n') for line in open('map.txt')]
for line in lines:
	tempList = line.split()
	targetPoints.append(int(tempList[0]))

# Holds number of ratios as defined by the map
totalTargets = int(len(targetPoints))

# Start video capture (webcam)
video = cv2.VideoCapture(0)

detections = 0
detectionIndices = []

while(True):
	ret, frame = video.read()
	cv2.imshow('Original video feed', frame)

	#Convert the frame to grayscale
	grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Activating Haar cascade classifier to detect faces
	faces = face_cascade.detectMultiScale(grayFrame, scaleFactor = 1.5, minNeighbors = 5)

	for(x, y, w, h) in faces :
		pillowImage = Image.fromarray(frame[y:y+h, x:x+w])
		#Resizing dimensions
		resizedHeight = 300
		resizedWidth = 300
		######
		faceCropped = np.array(pillowImage.resize((resizedHeight, resizedWidth), Image.ANTIALIAS))

		#Initialize dlib's rectangle to start plotting points over shape of the face
		dlibRect = dlib.rectangle(0, 0, resizedHeight, resizedWidth)
		shape = predictor(cv2.cvtColor(faceCropped, cv2.COLOR_BGR2GRAY), dlibRect)
		shapeCopy = shape
		shape = face_utils.shape_to_np(shape)

		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			cv2.imshow('Detected face', faceCropped)
			os.system('clear')

			baseLine = getDistance(shapeCopy, 28, 27)
			ratios = []

			for x in targetPoints:
				currentLine = getDistance(shapeCopy, x, 27)
				currentRatio = float(currentLine)/float(baseLine)
				ratios.append(currentRatio)

			foundFlag = 0

			for x in range(0, totalTargets):
				if(x % 3 == 0):
					print("\nP {}: {}\t\t".format(x+1, ratios[x])),
				else:
					print("P {}: {}\t\t".format(x+1, ratios[x])),					

			# Keras neural net
			npratios = []
			npratios.append(ratios)
			npratios = np.array(npratios)

			kerasOutput = model.predict(npratios)

			print("\nKERAS O/P: {}".format(kerasOutput))
			print("\nMAP: [Bobby, Omkar, Chinmay, Sumit, Arjun]")
			maxval = -1
			maxid = -1
			for x in range(0, len(kerasOutput[0])):
				if (kerasOutput[0][x] > maxval):
					maxval = kerasOutput[0][x]
					maxid = x

			print("\nMAX CONFIDENCE FOR INDEX : {}".format(maxid))
			exit(0)
			#
			'''
			detections = detections + 1
			detectionIndices.append(maxid)

			if(detections == 5):
				print(detectionIndices)
				exit(0)
			'''

	if cv2.waitKey(20) & 0xFF == ord('q') :
		break

video.release()
cv2.destroyAllWindows()

#g0d0d1f0g0h0i1j0b0e0e0e0b1d0c1d1b1f1f0f1g1g1g1g0f1
#g0d0d1f0g0h0i1j0b0e0e0e0b1d0c1d1b1f1f0f1g1g1g1g0f1
#g0d0d1f0g0h0i1j0b0e0e0e0b1d0c1d1b1f1f0f1g1g1g1g0f1
#g0d0d1f0g0g1i0i1b0d1e0e0b1d0c1d1b1f0f0f1g0g0g1f1f1

#s0i1l1p0s1u1z0b0d0l1m0m0f0j1h1k1e1p1p1q1t0t0t1r0r0
#s0i1l1p0s1v0y1a1d0l1m0l1f0j1h0k1e1p1p1q1t0t0t0r0r0