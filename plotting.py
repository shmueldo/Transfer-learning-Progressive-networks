import gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
from ModifiedTensorBoard import *
import os
from datetime import datetime
import time

if __name__ == '__main__':
    # for lr in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.009]:
    # for lr in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.009, 0.01]:
    # for lr in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0009, 0.001]:
    # for df in [0.9 ,0.95 ,0.99 ,0.995 ,0.9995]:
    # for lr in [0.0001, 0.0002, 0.0003,
    #             0.0004, 0.0005, 0.0006, 0.0007, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.009, 0.01]:
    # for lr in [0.00004, 0.00006, 0.00007 ,0.00009, 0.0001, 0.0002, 0.0004, 0.0005, 0.0006, 0.0007, 0.0009, 0.001, 0.003, 0.005, 0.007, 0.01]:      
    # for lr in [0.00005, 0.00007, 0.0001, 0.0002, 0.0003, 0.0004,
    #        0.0006, 0.0009, 0.001, 0.002, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03, 0.05]:
    # for lr in [0.00001, 0.00005, 0.00004, 0.00006, 0.00007 ,0.00009, 0.0001, 0.0002, 0.0004, 0.0005, 0.0006, 0.0007, 0.0009, 0.001, 0.003, 0.005, 0.007, 0.01]:
    # for lr in [ 2e-5, 3e-5, 5e-5, 5e-6, 7e-5]:
        # env_name = 'CartPole-v1'
        # env_name = 'Acrobot-v1' 
    env_name = 'MountainCarContinuous-v0'
    # mode = 'policy'
    mode = 'sv'
        # with open(os.getcwd() + '\simulationsQ2\{}\{}_sv_sim_lr={}_owner=CartPole-v1.npy'.format(env_name, env_name, lr), 'rb') as f:

    for lr in [0.0004, 0.00045, 0.0005, 0.0006, 0.0007, 0.0009,
               0.001, 0.002, 0.003, 0.004, 0.0045, 0.0055, 0.006, 0.007, 0.008, 0.01, 0.004, 0.0041,
               0.0042, 0.0043, 0.0044, 0.0046, 0.0047, 0.0048, 0.0049, 0.005,
               0.0051, 0.0052, 0.0053, 0.0054]: 
        with open(os.getcwd() +'\simulations\simulationsQ3\{}\{}_sv_sim_lr={}_policy=0.00045.npy'.format(env_name, env_name, lr), 'rb') as f:
            last_episode = np.load(f)
            rewards = np.load(f)
            mean_rewards = np.load(f)
            print('for lr: {}, algorithm converged after {} episodes with score of {}'.format(lr, len(mean_rewards), mean_rewards[-1]))
            losses = np.abs(np.load(f))
        # Custom tensorboard object
        tensorboard = ModifiedTensorBoard("{}_lr={}".format(mode, lr),
        log_dir="{}logs/{}/Q3_{}_lr_sim".format(os.getcwd() + r"/", env_name, mode))
        for episode in range(last_episode):
            #update tensor board step
            tensorboard.step = episode
            tensorboard.update_stats(episode_rewards = int(rewards[episode]), mean_reward= mean_rewards[episode], loss=losses[episode])