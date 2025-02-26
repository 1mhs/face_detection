import dlib
import numpy as np

# dlib의 정면 얼굴 탐지기 초기화
detector = dlib.get_frontal_face_detector()

# 얼굴 특징 추출기 초기화
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 알려진 얼굴 이미지 로드
known_image = dlib.load_rgb_image('hs.jpg')

# 알려진 얼굴 탐지
faces = detector(known_image)

# 알려진 얼굴 특징 추출
landmarks = predictor(known_image, faces[0])

# 얼굴 특징 벡터 추출
face_descriptor = np.array([
    landmarks.part(n).x for n in range(68)] +
    [landmarks.part(n).y for n in range(68)]
)

# 얼굴 특징 벡터 저장
np.save('known_face_descriptor.npy', face_descriptor)
