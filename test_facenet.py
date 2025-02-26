import cv2
import numpy as np
import tensorflow as tf

# FaceNet 모델 로드
model_path = 'C:/opencv/facenet_keras.h5'
model = tf.keras.models.load_model(model_path)

# 이미지 전처리
def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# 얼굴 임베딩 추출 함수
def get_face_embedding(image):
    preprocessed_image = preprocess_image(image)
    embedding = model.predict(preprocessed_image)
    return embedding[0]

video_capture = cv2.VideoCapture(0)

cascade_path = 'C:/opencv/haarcascade_frontalface_default.xml' 
face_cascade = cv2.CascadeClassifier(cascade_path)

# 거리 임계값을 설정
threshold = 0.4


embeddings = np.load('C:/opencv/embedding.npy', allow_pickle=True)


def distance(emb1, emb2):
    return np.sqrt(np.sum(np.square(emb1 - emb2)))


def find_closest_match(embedding):
    min_dist = float('inf')
    min_index = None
    for i in range(len(embeddings)):
        dist = distance(embedding, embeddings[i])
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return min_index, min_dist

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        embedding = get_face_embedding(face)
        index, dist = find_closest_match(embedding)

        # 거리가 일정 범위 이내인 경우 해당 인물의 이름을 출력
        if dist < threshold:
            name = 'son'  
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
