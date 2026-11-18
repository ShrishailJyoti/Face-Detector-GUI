import cv2
import sys

# Path to the Haar Cascade XML file
cascPath ="face_detector.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Initialize the video capture (0 is usually the built-in webcam)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale for the detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Save the current frame as an image file
    cv2.imwrite("face_det.png", frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
