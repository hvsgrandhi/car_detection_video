#importing openCv library
import cv2

#creating a variable for video
video = cv2.VideoCapture('video.mp4')

#pre-trained car classifier
classifier_file = 'car_detection.xml'

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#run forever until video stops
while True:

    #reading a frame from the video
    (read_successful, frame) = video.read()

    #safe coding
    if read_successful:
        #converting each frame into black and white(grayscale)
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    #draw rectangles around the cars
    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cv2.rectangle(image, (x n y corrdinates), (x+w n y+h coordinate), (color of the rectangle in form of bgr), (thickness of the rectangle))



    #displays the images with cars
    cv2.imshow('The name of the window', frame)

    #do not auto close the window(closses on any keypress)
    cv2.waitKey(1)

print("COmpletion of the code!")


