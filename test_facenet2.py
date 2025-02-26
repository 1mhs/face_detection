import cv2
import numpy as np
from tensorflow.keras.models import load_model

# FaceNet 모델 로드
model_path = 'C:/opencv/facenet_keras.h5'  # FaceNet 모델 파일 경로
model = load_model(model_path)

# FaceNet 모델을 사용하여 얼굴 임베딩 벡터 추출
def extract_face_embeddings(face_image):
    image_data = cv2.resize(face_image, (160, 160))  
    image_data = image_data[np.newaxis, :, :, :]  
    embeddings = model.predict(image_data)
    return embeddings

# 얼굴 검출을 위한 Haar Cascade 로드
cascade_path = 'C:/opencv/haarcascade_frontalface_default.xml'  # Haar Cascade 파일 경로
face_cascade = cv2.CascadeClassifier(cascade_path)

def face_recognition():
    cap = cv2.VideoCapture(0) 

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  

        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]  

            # 얼굴 임베딩 벡터 추출
            embeddings = extract_face_embeddings(face_image)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 

        cv2.imshow('Face Recognition', frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face_recognition()
