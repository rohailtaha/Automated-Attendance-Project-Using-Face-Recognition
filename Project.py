import cv2 as cv
from face_recognition.api import face_distance;
import numpy as np;
import face_recognition;
import os;
from datetime import datetime;

KNOWN_IMAGES_PATH = 'images';
images = [];
names = [];

for fileName in os.listdir(KNOWN_IMAGES_PATH):
  img = cv.imread(f'{KNOWN_IMAGES_PATH}/{fileName}');
  images.append(img);
  names.append(os.path.splitext(fileName)[0]);

# get face encodings of images. 
# NOTE: Each image should have only one face
def face_encodings(images):
  encodings = [];
  for img in images:
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB);
    encoding = face_recognition.face_encodings(img)[0];
    encodings.append(encoding);
  return encodings;  

def mark_attendance(name):
  with open('attendance.csv', 'r+') as f:

    def marked_attendance(name):
      myDataList = f.readlines();
      names = [];
      for line in myDataList:
        names.append(line.split(',')[0]);
      return 1 if name in names else 0;  

    if not marked_attendance(name):
      date = datetime.now().strftime('%H:%M:%S');
      f.writelines(f'\n{name},{date}')

encodingsKnown = face_encodings(images);
print('Encoding Complete');

VIDEO_PATH = 'input-images/video.mp4';
cap = cv.VideoCapture(VIDEO_PATH);

# Keep reading individual frames of video and detect and recognize face.
while 1:
  success, img = cap.read();
  # make image size small for quick operations
  imgSmall = cv.resize(img, (0,0), None, 0.25, 0.25);
  imgSmall = cv.cvtColor(imgSmall, cv.COLOR_BGR2RGB);

  faceLocationsCurrentFrame = face_recognition.face_locations(imgSmall);
  encodingsCurrentFrame =  face_recognition.face_encodings(imgSmall, faceLocationsCurrentFrame);

  for encoding, faceLocation in  zip(encodingsCurrentFrame, faceLocationsCurrentFrame):
    matches = face_recognition.compare_faces(encodingsKnown, encoding); 
    faceDistance = face_recognition.face_distance(encodingsKnown, encoding);
    matchIndex = np.argmin(faceDistance)
    # check if the the minimum face distance is actually a match or not
    if matches[matchIndex]:
      mark_attendance(names[matchIndex]);

      # Put rectangle around face and person name in image
      y1,x2,y2,x1 = faceLocation;
      y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4;
      cv.rectangle(img, (x1, y1),(x2,y2), (0,255,0), 2);
      cv.rectangle(img, (x1, y2-35),(x2,y2), (0,255,0), cv.FILLED);
      cv.putText(img, names[matchIndex], (x1+6, y2-6),cv.FONT_HERSHEY_COMPLEX, 1,  (255,255,255), 2);

  cv.imshow('img', img);
  cv.waitKey(1)


