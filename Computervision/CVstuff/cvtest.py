import cv2
import sys
import time

cascPath = 'haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
#video_capture.set(3,1280)
#video_capture.set(4,720)

fps_ctr = 0
fps = 0
t1 = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    frame = frame[120:360, 160:480, :]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20),
        maxSize=(50, 50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    """ Calculate fps """
    fps_ctr += 1

    if fps_ctr >= 5:
        t2 = time.time()
        period = t2-t1
        fps = 5./(t2-t1)
        t1 = t2
        fps_ctr = 0

    cv2.putText(frame, str(int(fps)) + " fps", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()