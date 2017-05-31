#!/usr/bin/python2.7
# encoding: utf-8

'''
Created on 2017年5月24日

@author: zenghao
'''

import cv2
from BrainDQN_Nature import BrainDQN
import numpy as np
from game import GameObject
from config import IMG_HEIGHT, IMG_WIDTH

# TRAIN = False
TRAIN = True
# preprocess raw image to img_size*img_size gray image
def preprocess(observation):
#     observation = cv2.resize(observation, (IMG_WIDTH, IMG_HEIGHT))
    observation = cv2.resize(observation, (IMG_HEIGHT, IMG_WIDTH))
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    return np.reshape(observation, (IMG_WIDTH, IMG_HEIGHT, 1))

def playGame():
    game = GameObject()
    game.train = TRAIN
    actions = 3
    brain = BrainDQN(actions, game)
    action0 = np.array([1, 0, 0])  # do nothing
    observation0, _ = game.frame_step(action0)
    observation0 = cv2.resize(observation0, (IMG_HEIGHT, IMG_WIDTH))
    observation0 = cv2.cvtColor(observation0, cv2.COLOR_BGR2GRAY)

    brain.setInitState(observation0)

    while True:
        action = brain.getAction()
        nextObservation, reward = game.frame_step(action)
        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation, action, reward)


if __name__ == '__main__':
    playGame()
