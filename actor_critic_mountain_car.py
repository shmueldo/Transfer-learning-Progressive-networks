import gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
from ModifiedTensorBoard import *
import os
from datetime import datetime
import time
import tensorflow_probability as tfp
import scipy.stats as ss

# optimized for Tf2
tf.disable_v2_behavior()

algorithm_name = "actor_critic"

# Actor
class PolicyNetwork:
    def __init__(self, state_size, action_size, env_action_size, learning_rate, net_name='policy_network',
                 fine_tuning = False, weights_owner_env = None):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(net_name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.float32, [env_action_size], name="action")
            self.A_t = tf.placeholder(tf.float32, name="discounted_advantage")
            self.I_factor = tf.placeholder(tf.float32, name="I_factor")
            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            
            if (fine_tuning):
                print("Loading weights from {}".format(weights_owner_env))
                layers_weights = {}
                for layer in ["W1", "b1", "W2", "b2"]:
                    with open(os.getcwd() + '\weights\{}\{}\{}.npy'.format(weights_owner_env, net_name, layer), 'rb') as f:
                        layers_weights[layer] = np.load(f)
                
                self.W1 = tf.get_variable("W1", initializer=tf.constant(layers_weights["W1"]))
                self.b1 = tf.get_variable("b1", initializer=tf.constant(layers_weights["b1"]))
                self.W2 = tf.get_variable("W2", initializer=tf.constant(layers_weights["W2"]))
                self.b2 = tf.get_variable("b2", initializer=tf.constant(layers_weights["b2"]))
           
            else:
                self.W1 = tf.get_variable("W1", [self.state_size, 64], initializer=tf2_initializer)
                self.b1 = tf.get_variable("b1", [64]                 , initializer=tf2_initializer)
                self.W2 = tf.get_variable("W2", [64, 32]             , initializer=tf2_initializer)
                self.b2 = tf.get_variable("b2", [32]                 , initializer=tf2_initializer)
            
            self.W3 = tf.get_variable("W3", [32, self.action_size], initializer=tf2_initializer)
            self.b3 = tf.get_variable("b3", [self.action_size]    , initializer=tf2_initializer)

            # Model flow
            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            # extract probability distribution from output
            self.actions_distribution = tf.squeeze(self.output[:, :2])
            self.mu = tf.clip_by_value(self.actions_distribution[0], -1, 1)
            self.sigma = tf.nn.softplus(self.actions_distribution[1]) + 1e-5
            
            self.output_action = tf.clip_by_value(tf.squeeze(tf.random.normal([1], self.mu, self.sigma, tf.float32, seed=1), axis=0), -1, 1)
            self.norm_dist = tf.distributions.Normal(self.mu, self.sigma)
            
            # Loss with log probability
            self.neg_log_prob = -tf.log(self.norm_dist.prob(self.action) + 1e-5)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.I_factor * self.A_t)
            if (fine_tuning):
                var_list = [self.W2, self.b2, self.W3, self.b3]
                # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=var_list)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            else:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

# Critic
class ValueNetwork:
    def __init__(self, state_size, learning_rate, net_name='value_network',
                 fine_tuning = False, weights_owner_env = None):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(net_name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.A_t = tf.placeholder(tf.float32, name="discounted_advantage")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.I_factor = tf.placeholder(tf.float32, name="I_factor")
            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)

            if (fine_tuning):
                print("Loading weights from {}".format(weights_owner_env))
                layers_weights = {}
                for layer in ["W1", "b1", "W2", "b2"]:
                    with open(os.getcwd() + '\weights\{}\{}\{}.npy'.format(weights_owner_env, net_name, layer), 'rb') as f:
                        layers_weights[layer] = np.load(f)
                
                self.W1 = tf.get_variable("W1", initializer=tf.constant(layers_weights["W1"]))
                self.b1 = tf.get_variable("b1", initializer=tf.constant(layers_weights["b1"]))
                self.W2 = tf.get_variable("W2", initializer=tf.constant(layers_weights["W2"]))
                self.b2 = tf.get_variable("b2", initializer=tf.constant(layers_weights["b2"]))
           
            else:
                self.W1 = tf.get_variable("W1", [self.state_size, 64], initializer=tf2_initializer)
                self.b1 = tf.get_variable("b1", [64]                 , initializer=tf2_initializer)
                self.W2 = tf.get_variable("W2", [64, 16]             , initializer=tf2_initializer)
                self.b2 = tf.get_variable("b2", [16]                 , initializer=tf2_initializer)
            
            self.W3 = tf.get_variable("W3", [16, 1], initializer=tf2_initializer)
            self.b3 = tf.get_variable("b3", [1], initializer=tf2_initializer)
            
            # Model flow 
            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)
                        
            # Mean squared error loss
            self.mse = tf.losses.mean_squared_error(predictions=self.output, labels=self.R_t)
            self.loss = tf.reduce_mean(self.mse * self.I_factor)
            if (fine_tuning):
                var_list = [self.W3, self.b3]
                # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=var_list)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            else:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

def run(discount_factor: float, policy_learning_rate: float ,sv_learning_rate: float,
        env_name: str, desired_goal=475, action_space= [0, 1, 2], max_steps = 501,
        fine_tuning = False, weights_owner_env = None, save_weights = False):
    # Assign seed for repetitive randomness
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    
    # Initialize general settings    
    rewards, mean_rewards, losses = [], [], []
    discount_factor = discount_factor
    policy_learning_rate = policy_learning_rate
    sv_learning_rate = sv_learning_rate
    render = False
    max_episodes = 500
    unified_state_size = 6
    unified_action_size = 3

    
    # Initialize environment parameters    
    env = gym.make(env_name)
    env.seed(SEED)
    try:
        env_action_size = env.action_space.n
    except AttributeError:
        env_action_size = 1
    env_state_size = env.observation_space.shape[0]
    max_steps = max_steps     # defined manually

    # Initialize the policy and the state-value network
    tf.reset_default_graph()
    policy = PolicyNetwork(unified_state_size, unified_action_size, env_action_size, policy_learning_rate,
                           fine_tuning=fine_tuning, weights_owner_env=weights_owner_env)
    state_value = ValueNetwork(unified_state_size, sv_learning_rate, fine_tuning=fine_tuning, weights_owner_env=weights_owner_env)
    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solved = False
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0
        since = time.time()
        for episode in range(max_episodes):
            state = env.reset()
            
            # adjust state size into the unified one
            state = np.pad(state, (0, unified_state_size - env_state_size))
            state = state.reshape([1, unified_state_size])
            I_factor = 1

            for step in range(max_steps):
                # Sample action from action distribution
                output_action, mu, sigma = sess.run([policy.output_action, policy.mu, policy.sigma], {policy.state: state})
                
                modified_action = np.array(output_action, dtype=np.float32, ndmin=1)
                next_state, reward, done, _ = env.step(modified_action)
                episode_rewards[episode] += reward

                ## Modified reward section
                # if (next_state[0] > 0.42):
                #     reward += 30000/step
                
                # if (next_state[0] >= 0.45):
                #     reward += (1000 - step) 
                # elif (next_state[0] > 0.2):
                #     reward += 5
                # elif (next_state[0] > -0.05):
                #     reward += 1

                # true_state = np.abs(np.cos(np.pi/3.) + next_state[0])
                # reward += -(1. - true_state)

                # true_state = np.abs(0.45 + next_state[0])
                # reward += -(1. - true_state)
                
                ## Acceleration manner reward
                reward = reward + 100 * 0.7 * (abs(next_state[1]) - abs(state[0][1]))  # 10 * self.gamma * (abs(next_obs[1]) - abs(obs[1]))
                
                next_state = np.pad(next_state, (0, unified_state_size - env_state_size))  # adjust state size
                next_state = next_state.reshape([1, unified_state_size])

                if render:
                    env.render()

                # Calculate state-value output for current state
                feed_dict = {state_value.state: state}
                value_current_state = sess.run(state_value.output, feed_dict)

                # Calculate state-value output for next state
                feed_dict = {state_value.state: next_state}
                value_next_state = sess.run(state_value.output, feed_dict)
 
                # calculate advantage
                if done:
                    target = reward
                else:
                    target = reward + discount_factor * value_next_state
                advantage = target - value_current_state
                
                # Update the state_value network weights
                feed_dict = {state_value.state: state, state_value.A_t: advantage,
                             state_value.R_t: target, state_value.I_factor: I_factor}
                _, loss_state = sess.run([state_value.optimizer, state_value.loss], feed_dict)

                # Update the policy network weights
                feed_dict = {policy.state: state, policy.A_t: advantage,
                             policy.action: modified_action, policy.I_factor : I_factor}
                _, loss_policy = sess.run([policy.optimizer, policy.loss], feed_dict)

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                    print("Episode {}, steps {}, Reward: {} Average over 100 episodes: {}".format(episode, step, round(episode_rewards[episode], 2), round(average_rewards, 2)))
                    if average_rewards > desired_goal and average_rewards !=0:
                        print(' Solved at episode: ' + str(episode))
                        time_elapsed = time.time() - since
                        print("Algorithm {} for {} converged after {} seconds".format(algorithm_name, env_name, time_elapsed))
                        solved = True
                    break
                
                I_factor *= discount_factor
                state = next_state

            if solved:
                if (save_weights):
                    # save all layers besides the final one
                    policy_weights      = sess.run([policy.W1, policy.b1, policy.W2, policy.b2])
                    state_value_weights = sess.run([state_value.W1, state_value.b1, state_value.W2, state_value.b2])
                    for i, layer in enumerate(["W1", "b1", "W2", "b2"]):
                        with open(os.getcwd() + '\weights\{}\{}\{}.npy'.format(env_name, "policy_network", layer), 'wb') as f:
                            np.save(f, policy_weights[i])
                        with open(os.getcwd() + '\weights\{}\{}\{}.npy'.format(env_name, "value_network", layer), 'wb') as f:
                            np.save(f, state_value_weights[i])
                break
    
            rewards.append(episode_rewards[episode])
            mean_rewards.append(average_rewards)
            losses.append(loss_policy)
    return episode, rewards, mean_rewards, losses

if __name__ == '__main__':
    SEED = 42
    env_name = 'MountainCarContinuous-v0'
    
    optimal_sv_lr = {'CartPole-v1' : 0.003, 'Acrobot-v1' : 0.0005, 'MountainCarContinuous-v0' : 5e-5}
    optimal_policy_lr = {'CartPole-v1' : 0.003, 'Acrobot-v1' : 0.0005, 'MountainCarContinuous-v0' : 0.0002}
    optimal_df = {'CartPole-v1' : 0.99, 'Acrobot-v1' : 0.99, 'MountainCarContinuous-v0' : 0.99}
    env_goal      = {'CartPole-v1' : 475, 'Acrobot-v1'  : -90, 'MountainCarContinuous-v0' : 75}
    max_steps      = {'CartPole-v1' : 501, 'Acrobot-v1'  : 501, 'MountainCarContinuous-v0' : 1000}
    actions_space = {'CartPole-v1' : [0, 1], 'Acrobot-v1' : [0, 1, 2],
                     'MountainCarContinuous-v0' : np.linspace(-1, 1, 32)}
    
    # TODO: Dont forget to add 0.00001, 0.00005 to list when plotting
    # for lr in [0.00001, 0.00005, 0.00004, 0.00006, 0.00007 ,0.00009, 0.0001, 0.0002, 0.0004, 0.0005, 0.0006, 0.0007, 0.0009, 0.001, 0.003, 0.005, 0.007, 0.01]:                                                    
    # for lr in [0.00001, 0.00005, 0.00004, 0.00006, 0.00007 ,0.00009, 0.0001, 0.0002, 0.0004, 0.0005, 0.0006, 0.0007, 0.0009, 0.001, 0.003, 0.005, 0.007, 0.01]:                                                    
    for lr in [0.00005]:
    # TODO: Run it with sv = 5e-5                                                  
        last_episode, rewards, mean_rewards, losses = run(env_name=env_name,
                                                            discount_factor=optimal_df[env_name],
                                                            policy_learning_rate=optimal_policy_lr[env_name],
                                                            sv_learning_rate=optimal_sv_lr[env_name], 
                                                            desired_goal=env_goal[env_name],
                                                            action_space=actions_space[env_name],
                                                            max_steps=max_steps[env_name],
                                                            fine_tuning = False, weights_owner_env = 'Acrobot-v1',
                                                            save_weights = True)
        with open(os.getcwd() + '\simulations\{}\{}_optimal_simulation.npy'.format(env_name, env_name), 'wb') as f:
            np.save(f, last_episode)
            np.save(f, rewards)
            np.save(f, mean_rewards)
            np.save(f, losses)