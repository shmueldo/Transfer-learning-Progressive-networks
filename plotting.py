import gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
from ModifiedTensorBoard import *
import os
from datetime import datetime
import time

if __name__ == '__main__':
    # for lr in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0009, 0.001]:
    for df in [0.9 ,0.95 ,0.99 ,0.995 ,0.9995]:
        algorithm_name = "reinforce_wt_baseline"
        # with open('sv_lr_sim_{}_{}.npy'.format(algorithm_name, lr), 'rb') as f:
        with open('df_sim_{}_{}.npy'.format(algorithm_name, df), 'rb') as f:
            last_episode = np.load(f)
            rewards = np.load(f)
            mean_rewards = np.load(f)
            losses = np.abs(np.load(f))
        
        algorithm_name = "reinforce_wt_baseline_df={}".format(df)
        # Custom tensorboard object
        tensorboard = ModifiedTensorBoard(algorithm_name,
        log_dir="{}logs/re_wt_b_df_sim".format(os.getcwd() +r"/"))
        for episode in range(last_episode):
            #update tensor board step
            tensorboard.step = episode
            tensorboard.update_stats(episode_rewards = int(rewards[episode]), mean_reward= mean_rewards[episode], loss=losses[episode])