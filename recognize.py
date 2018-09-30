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
import face_recognition

#Set name of new user
currentUser = raw_input("Enter name of current user : ")
model = keras.models.load_model(currentUser + '.model')

#Importing Haar cascade and DLIB's facial landmarks detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture (webcam)
video = cv2.VideoCapture(0)

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

		start_time = time.time()
		encoded = face_recognition.face_encodings(faceCropped)
		print("--- Encoding time: %s seconds ---" % (time.time() - start_time))

		if(not len(encoded) == 0):

			# Keras neural net
			npratios = []
			npratios.append(encoded[0])
			npratios = np.array(npratios)

			start_time = time.time()
			kerasOutput = model.predict(npratios)

			print("--- Detection time: %s seconds ---" % (time.time() - start_time))
			print("\nKERAS O/P: {}".format(kerasOutput))
			
			maxValue = kerasOutput[0][0]

			for value in kerasOutput[0]:
				if(maxValue < value):
					maxValue = value

			if(maxValue == kerasOutput[0][0] and maxValue > 0.99):
				print("\nCONFIDENCE : {}".format(kerasOutput[0][0]*100))
				exit(0)

	

	if cv2.waitKey(20) & 0xFF == ord('q') :
		break

video.release()
cv2.destroyAllWindows()