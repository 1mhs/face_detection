import pygame
import dlib
import cv2
import numpy as np
import time


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


LANE_WIDTH = 200
LANE_START_X = WINDOW_WIDTH // 2 - LANE_WIDTH // 2


CAR_WIDTH = 80
CAR_HEIGHT = 160
CAR_START_X = LANE_START_X + LANE_WIDTH // 2 - CAR_WIDTH // 2
CAR_START_Y = WINDOW_HEIGHT - CAR_HEIGHT - 10


LANE_COLOR = (30, 80, 30)
BACKGROUND_COLOR = (100, 100, 100)
DASH_COLOR = (255, 255, 255)
DASH_WIDTH = 10
DASH_HEIGHT = 40
DASH_GAP = 40
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("자동차 주행 게임")

clock = pygame.time.Clock()

car_image = pygame.image.load("car.jpeg")
car_rect = car_image.get_rect()
car_rect.x = CAR_START_X
car_rect.y = CAR_START_Y

def draw_dashed_line():
    dash_y = 0
    while dash_y < WINDOW_HEIGHT:
        dash_rect = pygame.Rect(WINDOW_WIDTH // 2 - DASH_WIDTH // 2, dash_y, DASH_WIDTH, DASH_HEIGHT)
        pygame.draw.rect(window, DASH_COLOR, dash_rect)
        dash_y += DASH_HEIGHT + DASH_GAP

car_speed = 5

# 정면 얼굴 탐지
detector = dlib.get_frontal_face_detector()
# 68개 얼굴 특징점을 찾는 모델
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# 미리 저장된 얼굴 데이터 불러오기
known_face_descriptor = np.load('known_face_descriptor.npy')


cap = cv2.VideoCapture(0)
cv2.namedWindow("Webcam Feed")

running = True
move_car = False  
last_match_time = time.time()

while running:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        face_descriptor = np.array([
            landmarks.part(n).x for n in range(68)] +
            [landmarks.part(n).y for n in range(68)]
        )


        distance = np.linalg.norm(face_descriptor - known_face_descriptor)
        similarity = 1 / (1 + distance) 

        # 유사도가 일정값 이상인 경우에만 자동차를 움직임
        if similarity > 0.00045:
            move_car = True
            last_match_time = time.time()
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, "hyoseok", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, "Not Match", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 얼굴 유사도에 따라 자동차를 움직임
    if move_car:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and car_rect.x > LANE_START_X:
            car_rect.x -= car_speed
        if keys[pygame.K_RIGHT] and car_rect.x < LANE_START_X + LANE_WIDTH - CAR_WIDTH:
            car_rect.x += car_speed
        if keys[pygame.K_UP] and car_rect.y > 0:
            car_rect.y -= car_speed
        if keys[pygame.K_DOWN] and car_rect.y < WINDOW_HEIGHT - CAR_HEIGHT:
            car_rect.y += car_speed
    else:
        if time.time() - last_match_time < 3:
            move_car = False

    window.fill(BACKGROUND_COLOR)

 
    pygame.draw.rect(window, LANE_COLOR, (0, 0, LANE_START_X, WINDOW_HEIGHT))
    pygame.draw.rect(window, LANE_COLOR, (LANE_START_X + LANE_WIDTH, 0, WINDOW_WIDTH - (LANE_START_X + LANE_WIDTH), WINDOW_HEIGHT))

 
    draw_dashed_line()

    window.blit(car_image, car_rect)


    if move_car:
        pygame.draw.rect(window, GREEN, car_rect, 2)
    else:
        pygame.draw.rect(window, RED, car_rect, 2)

    pygame.display.flip()
    clock.tick(60)

 
    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
