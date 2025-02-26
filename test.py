import numpy as np

# 얼굴 특징 벡터 로드
face_descriptor = np.load('known_face_descriptor.npy')

# 특징 벡터 출력
print('[', end='')
print(*face_descriptor, sep=', ', end='')
print(']')
