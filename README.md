# Face-Recognition

How to use it?

First of all install all the requirements.
	pip install -r requirements.txt

Save the following file as 'haarcascade_frontalface_default.xml':
	https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

Download 'shape_predictor_68_face_landmarks.dat' from:
	http://dlib.net/files/

	Run train.py first to create a model for entered username.
	Then run recognize.py