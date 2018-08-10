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

#Basically finds distance between 2 points
def getDistance(tempshape, point1, point2):
	point1x = tempshape.part(point1).x
	point1y = tempshape.part(point1).y
	point2x = tempshape.part(point2).x
	point2y = tempshape.part(point2).y
				
	dx = (point2x - point1x) ** 2
	dy = (point2y - point1y) ** 2
	distance = math.sqrt(dx + dy)
	return distance


#Importing Haar cascade and dlib's facial landmarks detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#Precision in %
precision = 0.80

ratios = []
names = []
parameters = []
currentParameters = []
currentRatio = []

for i in range(0, 66):
	currentParameters.append(0.0)
	currentRatio.append(0.0)

#Read dataStore.txt
lines = [line.rstrip('\n') for line in open('dataStore.txt')]
for line in lines:
	tempList = line.split()
	names.append(tempList[0])
	tempList.remove(tempList[0])
	parameters.append(tempList)

for i in range(0, len(parameters)):
	newList = []
	ratios.append(newList)

	for x in range(0, 65):
		ratios[i].append(float(parameters[i][x])/float(parameters[i][x+1]))

#print(ratios)
#print(parameters)
#print(names)
records = int(len(names))
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

		#Initialize dlib's rectangle to start plotting points over shape of the face
		dlibRect = dlib.rectangle(0, 0, resizedHeight, resizedWidth)
		shape = predictor(cv2.cvtColor(faceCropped, cv2.COLOR_BGR2GRAY), dlibRect)
		shapeCopy = shape
		shape = face_utils.shape_to_np(shape)

		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			cv2.imshow('Detected face', faceCropped)
			os.system('clear')
			#We have the face in faceCropped
			#We have to check if it matches to our datastore
			#We check it with each record using individual threads

			for z in range(0, 66):
				tempValue = getDistance(shapeCopy, 67, z)
				currentParameters.remove(currentParameters[z])
				currentParameters.insert(z, tempValue)

			for count in range(0, records):
				foundFlag = 0
				os.system('clear')
				print("TEST FOR : {}\n".format(names[count]))
				for x in range(0, 65):
					if(currentParameters[x+1] == 0.0):
						break

					currentRatio = currentParameters[x]/currentParameters[x+1]
						
					if(currentRatio > ratios[count][x]*(1-(1 - precision)) and currentRatio < ratios[count][x]*(1+(1 - precision))):
						cv2.line(faceCropped, (shapeCopy.part(67).x, shapeCopy.part(67).y), (shapeCopy.part(x).x, shapeCopy.part(x).y), (0, 255, 0), 1)
						foundFlag = foundFlag + 1
						if(x % 3 == 0):
							print("\nP {}: {}\t[TRUE]\t".format(x+1, currentRatio)),
						else:
							print("P {}: {}\t[TRUE]\t".format(x+1, currentRatio)),					
					else:
						cv2.line(faceCropped, (shapeCopy.part(67).x, shapeCopy.part(67).y), (shapeCopy.part(x).x, shapeCopy.part(x).y), (0, 0, 255), 1)

						if(x % 3 == 0):
							print("\nP {}: {}\t\t".format(x+1, currentRatio)),
						else:
							print("P {}: {}\t\t".format(x+1, currentRatio)),

						break
				os.system('clear')


				if (foundFlag == 65):
					os.system('clear')
					print("MATCH FOUND : {}".format(names[count]))
					exit(0)


	if cv2.waitKey(20) & 0xFF == ord('q') :
		break

video.release()
cv2.destroyAllWindows()