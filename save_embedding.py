import tensorflow as tf
import numpy as np
import cv2

# FaceNet 모델 로드
model_path = 'C:/opencv/facenet_keras.h5'
model = tf.keras.models.load_model(model_path)

# 이미지 전처리 함수
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

image = cv2.imread('C:/opencv/hs.jpg')

embedding = get_face_embedding(image)

np.save('C:/opencv/embedding.npy', embedding)
