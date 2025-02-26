import pygame
import random


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

running = True
move_car = False 

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                move_car = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a:
                move_car = False


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


    window.fill(BACKGROUND_COLOR)

    pygame.draw.rect(window, LANE_COLOR, (0, 0, LANE_START_X, WINDOW_HEIGHT))
    pygame.draw.rect(window, LANE_COLOR, (LANE_START_X + LANE_WIDTH, 0, WINDOW_WIDTH - (LANE_START_X + LANE_WIDTH), WINDOW_HEIGHT))

    draw_dashed_line()

    window.blit(car_image, car_rect)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
