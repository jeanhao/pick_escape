#!/usr/bin/python2.7
# encoding: utf-8
import pygame
import sys
import random
from collections import deque
from pygame.locals import *  # @UnusedWildImport


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


FPS = 120
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
OBJECT_HEIGHT = 50
OBJECT_WIDTH = 50
SHORT_SCREENT_HEIGHT = 250

PLAYER_POS_Y = SCREEN_HEIGHT - OBJECT_HEIGHT
OBJECT_SIZE = (OBJECT_WIDTH, OBJECT_HEIGHT)
MAX_PLAYER_POS = int(SCREEN_WIDTH / OBJECT_WIDTH) - 3

STONE_INIT_POS_Y = 0  # 石头初始化位置
STONE_MAX_SIZE = 10
STONE_POS_X = [OBJECT_WIDTH * i for i in range(STONE_MAX_SIZE)]
# STONE_SIZE = [3, 5, 7]
STONE_SIZE = 8
STONE_VALUE = [1, 2, 3, 5, 10, -10, -5, -2, -1]
STONE_UPDATE_DISTANCE = 150  # 两层石头间的距离 + 100
STONE_SPEED = 5

MENU_LINE_X = SCREEN_WIDTH - 2 * OBJECT_WIDTH
SCORE_TEXT_X = SCREEN_WIDTH - 2 * OBJECT_WIDTH + 10
TEXT_HEIGHT_FLAG = 40
SCORE_SPEED_RECORD_TIME_INTERVAL = 1000  # 每两秒记录一次
SAVE_SPPED_FLAG = 100  # 保存次数


class GameObject(object):

    def __init__(self):
        pygame.init()
        self.score_font = pygame.font.SysFont("arial", 20)
        self.com_init()
        self.fps_clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('pick_escape')

    def com_init(self):
        self.init()
        self.score = 0
        # 用于统计速度
        self.last_score = deque([0])
        self.last_score_time = deque([0])
        self.speed = 0
        self.save_spped_index = 1

    def init(self, templated=False):
        self.play = False
        self.player_pos = 1
        self.stones = deque()
        if templated:
            for i in range(2, -1, -1):
                self.stones.append(self.gen_stones(50 + 150 * i))

    def welcome(self):
        self.screen.fill(BLACK)
        score_surface = self.score_font.render("Click space to start game", True, WHITE)
        self.screen.blit(score_surface, ((SCREEN_WIDTH - score_surface.get_width()) / 2,
                                     (SCREEN_HEIGHT - score_surface.get_height()) / 2))
        pygame.display.update()
        self.com_init()  # 更彻底地初始化，包括得分
        while not self.play:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_UP):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        self.play = True
            self.fps_clock.tick(FPS)  # 一帧后处理下个事件
        # 初始化游戏数据
        self.start()

    def start(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_UP):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_LEFT:
                        self.player_pos = max(self.player_pos - 1, 0)
                    elif event.key == K_RIGHT:
                        self.player_pos = min(self.player_pos + 1, MAX_PLAYER_POS)

            reward = self.update_stones()
            if reward == -1:  # 游戏结束
                self.score -= 1
                self.welcome()
            else:
                self.score += reward

            # 更新画面
            self.update_score_speed()
            self.update_screen()
            self.fps_clock.tick(FPS)  # 一帧后处理下个事件

    def update_screen(self):
        # 重绘背景
        self.screen.fill(BLACK)
        # 画右边部分，目前有分数和速度
        pygame.draw.line(self.screen, WHITE, (MENU_LINE_X, 0), (MENU_LINE_X, SCREEN_HEIGHT))
        score_text_surface = self.score_font.render("score", True, WHITE)
        self.screen.blit(score_text_surface, (SCORE_TEXT_X , 0))
        score_surface = self.score_font.render(str(self.score), True, WHITE)
        self.screen.blit(score_surface, (SCORE_TEXT_X, TEXT_HEIGHT_FLAG))

        score_text_surface = self.score_font.render("speed", True, WHITE)
        self.screen.blit(score_text_surface, (SCORE_TEXT_X, TEXT_HEIGHT_FLAG * 2))
        score_surface = self.score_font.render(str(self.speed), True, WHITE)
        self.screen.blit(score_surface, (SCORE_TEXT_X, TEXT_HEIGHT_FLAG * 3))

        # 画主角
        pygame.draw.rect(self.screen, WHITE, Rect((self.player_pos * OBJECT_WIDTH, PLAYER_POS_Y), OBJECT_SIZE))
        # 画石头
        for stones_row in self.stones:
            for stone_pos in stones_row:
                pygame.draw.rect(self.screen, WHITE, Rect(stone_pos, OBJECT_SIZE))
        pygame.display.update()

    def gen_stones(self, pos_y=STONE_INIT_POS_Y):  # 随机生成两个位置
        return [[pos, pos_y] for pos in random.sample(STONE_POS_X, STONE_SIZE)]

    def update_stones(self):
        # 看是否需要产生新的石头
        if not self.stones:  # 如果没有石头
            self.stones.append(self.gen_stones())
        else:
            # 获取最后一组stone的第一个的其y坐标
            last_stone_y = self.stones[-1][0][1]
            if last_stone_y > STONE_UPDATE_DISTANCE:
                self.stones.append(self.gen_stones())

        # 更新所有石头距离
        for row in range(len(self.stones)):
            for col in range(STONE_SIZE):
                self.stones[row][col][1] += STONE_SPEED  # 第col块石头y坐标

        # 检查第一组石头是否和player发生碰撞
        stone_y = self.stones[0][0][1]

        reward = 0
        if SCREEN_HEIGHT >= stone_y > SCREEN_HEIGHT - OBJECT_HEIGHT * 2 :  # 检查石头是否和用户发生碰撞
            for col in range(STONE_SIZE):
                if self.stones[0][col][0] / OBJECT_WIDTH == self.player_pos:
                    reward = -1
                    break
        elif stone_y > SCREEN_HEIGHT:  # 已经过了，给用户加分
            reward = 1
            self.stones.popleft()  # 清空石头

        return reward

    # 下面是训练相关
    def frame_step(self, input_actions):
        pygame.event.pump()

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: left
        # input_actions[2] == 1: right
        if input_actions[1] == 1:
            self.player_pos = max(self.player_pos - 1, 0)
        elif input_actions[2] == 1:
            self.player_pos = min(self.player_pos + 1, MAX_PLAYER_POS)

        reward = self.update_stones()

        self.score += reward
        if reward == -1:  # 发生了碰撞
            self.init(templated=True)

        # 更新速度
        self.update_score_speed()
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())[:MENU_LINE_X, SHORT_SCREENT_HEIGHT:]
        self.update_screen()

        self.fps_clock.tick(FPS)
        return image_data, reward

    def update_score_speed(self):  # 更新速度
        if pygame.time.get_ticks() - self.last_score_time[-1] > SCORE_SPEED_RECORD_TIME_INTERVAL:  # 超过记录长度，可以开始记录
            if(len(self.last_score) >= 10):
                self.speed = (self.score - self.last_score.popleft()) / (pygame.time.get_ticks() - self.last_score_time.popleft()) * 1000
            self.last_score_time.append(pygame.time.get_ticks())
            self.last_score.append(self.score)
            if self.save_spped_index >= SAVE_SPPED_FLAG:
                self.save_spped_flag = 1
                with open('speeds_file.txt', 'a') as f:
                    f.write("time:%s, score:%5f, speed:%5f\n" % (str(pygame.time.get_ticks()), self.score, self.speed))
            else:
                self.save_spped_index += 1


if __name__ == '__main__':
        GameObject().welcome()
