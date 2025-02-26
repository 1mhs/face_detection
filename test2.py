import dlib
import cv2
import numpy as np


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


known_face_descriptor = np.load('known_face_descriptor.npy')
print("Known Face Descriptor:")
print(np.array2string(known_face_descriptor, separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}, prefix='[', suffix=']'))


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        face_descriptor = np.array([
            landmarks.part(n).x for n in range(68)] +
            [landmarks.part(n).y for n in range(68)]
        )
        
        print("Current Face Descriptor:")
        print(np.array2string(face_descriptor, separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}, prefix='[', suffix=']'))
        
        similarity = np.linalg.norm(face_descriptor - known_face_descriptor)
        

        if similarity > 0.5:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, "Match", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, "Not Match", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    

    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
