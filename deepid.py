import cv2
import dlib
import numpy as np

def calculate_similarity(embedding1, embedding2):
    distance = np.sqrt(np.sum((embedding1 - embedding2) ** 2))
    similarity = 1.0 / (1.0 + distance)
    return similarity


face_detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor('C:/opencv/shape_predictor_68_face_landmarks.dat')


face_recognizer = dlib.face_recognition_model_v1('C:/opencv/dlib_face_recognition_resnet_model_v1.dat')

image1 = cv2.imread('C:/opencv/son7.jpg')
image2 = cv2.imread('C:/opencv/son8.jpg')


resized_image1 = cv2.resize(image1, (500, 500))
resized_image2 = cv2.resize(image2, (500, 500))

gray1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)

faces1 = face_detector(gray1)
faces2 = face_detector(gray2)

embedding1 = None
embedding2 = None

for face in faces1:
    landmarks = shape_predictor(gray1, face)
    embedding1 = np.array(face_recognizer.compute_face_descriptor(resized_image1, landmarks))

    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(resized_image1, (x, y), 2, (0, 255, 0), -1)

for face in faces2:
    landmarks = shape_predictor(gray2, face)
    embedding2 = np.array(face_recognizer.compute_face_descriptor(resized_image2, landmarks))

    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(resized_image2, (x, y), 2, (0, 255, 0), -1)

if embedding1 is not None and embedding2 is not None:
    similarity = calculate_similarity(embedding1, embedding2)
    print("유사도:", similarity)

cv2.imshow("Image 1", resized_image1)
cv2.imshow("Image 2", resized_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
