import gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
from ModifiedTensorBoard import *
import os
from datetime import datetime
import time


# optimized for Tf2
tf.disable_v2_behavior()

algorithm_name = "actor_critic"

def get_model_layers(state, desired_env, net_name):
    if (desired_env in ['Acrobot-v1', 'MountainCarContinuous-v0']):
        # assign model weights
        weights = {}
        for layer in ["W1", "b1", "W2", "b2"]:
            with open(os.getcwd() + '\weights\{}\{}\{}.npy'.format(desired_env, net_name, layer), 'rb') as f:
                weights[layer] = tf.get_variable(desired_env + "_" + layer, initializer=tf.constant(np.load(f)))

        # assign model layers
        layers = {}
        layers["h1"] = tf.nn.relu(tf.add(tf.matmul(state, weights["W1"]), weights["b1"]))
        layers["h2"] = tf.nn.relu(tf.add(tf.matmul(layers["h1"], weights["W2"]), weights["b2"]))

    elif desired_env.startswith("CartPole-v1"):
        net_weights = {"policy_network": ["W1", "b1"], "value_network": ["W1", "b1", "W2", "b2"]}
        # assign model weights
        weights = {}
        for layer in net_weights[net_name]:
            with open(os.getcwd() + '\weights\{}\{}\{}.npy'.format(desired_env, net_name, layer), 'rb') as f:
                weights[layer] = tf.get_variable(desired_env + "_" + layer, initializer=tf.constant(np.load(f)))

        # assign model layers
        layers = {}
        layers["h1"] = tf.nn.relu(tf.add(tf.matmul(state, weights["W1"]), weights["b1"]))
        if net_name.startswith("value_network"):
            layers["h2"] = tf.nn.relu(tf.add(tf.matmul(layers["h1"], weights["W2"]), weights["b2"]))
    return weights, layers

class PolicyNetwork:
    def __init__(self, state_size, action_size, env_action_size, learning_rate,
                 net_name='policy_network', env_name = 'CartPole-v1'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        self.environments = ['Acrobot-v1', 'MountainCarContinuous-v0', 'CartPole-v1']
        with tf.variable_scope(net_name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.A_t = tf.placeholder(tf.float32, name="discounted_advantage")
            self.I_factor = tf.placeholder(tf.float32, name="I_factor")
            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            
            if env_name.startswith('MountainCarContinuous-v0'):
                self.action = tf.placeholder(tf.float32, [env_action_size], name="action")
                trainable_list = []
                cartpole_weights, cartpole_layers = get_model_layers(self.state, 'CartPole-v1', net_name)
                acrobot_weights, acrobot_layers = get_model_layers(self.state, 'Acrobot-v1', net_name)
                
                # First floor weights
                self.W1_mc = tf.get_variable("W1_mc", [self.state_size, 64], initializer=tf2_initializer) 
                trainable_list.append(self.W1_mc)
                
                self.b1_mc = tf.get_variable("b1_mc", [64]                 , initializer=tf2_initializer)
                trainable_list.append(self.b1_mc)
                
                # First floor layers
                self.h1_mc = tf.nn.relu(tf.add(tf.matmul(self.state, self.W1_mc), self.b1_mc)) # MountainCar first hidden
                self.h1_cp = cartpole_layers["h1"]  # Cartpole first hidden
                self.h1_ac = acrobot_layers["h1"]   # Acrobot first hidden
                
                # Second floor weights
                self.W2_mc_mc = tf.get_variable("W2_mc_mc", [64, 32], initializer=tf2_initializer)  # h1 MountainCar to h2 MountainCar weights
                trainable_list.append(self.W2_mc_mc)
                self.W2_cp_mc = tf.get_variable("W2_cp_mc", [64, 32], initializer=tf2_initializer)  # h1 Cartpole to h2 MountainCar weights
                trainable_list.append(self.W2_cp_mc)
                self.W2_ac_mc = tf.get_variable("W2_ac_mc", [64, 32], initializer=tf2_initializer)  # h1 Acrobot to h2 MountainCar weights
                trainable_list.append(self.W2_ac_mc)
                self.b2_mc_mc = tf.get_variable("b2_mc_mc", [32]    , initializer=tf2_initializer)  # h1 to h2 bias MountainCar
                trainable_list.append(self.b2_mc_mc)
                self.W2_cp_ac = tf.get_variable("W2_cp_ac", [64, 32], initializer=tf2_initializer)  # h1 Cartpole to h2 Acrobot weights
                trainable_list.append(self.W2_cp_ac)
                
                self.W2_ac_ac = acrobot_weights["W2"]   # h1 Acrobot to h2 Acrobot weight
                self.b2_ac_ac = acrobot_weights["b2"]   # h1 Acrobot to h2 Acrobot weight
                
                # Second floor layers         
                h1_mc_W2_mc_mc = tf.matmul(self.h1_mc, self.W2_mc_mc)
                h1_cp_W2_cp_mc = tf.matmul(self.h1_cp, self.W2_cp_mc)
                h1_ac_W2_ac_mc = tf.matmul(self.h1_ac, self.W2_ac_mc)                
                self.h2_mc = tf.nn.relu(tf.add(tf.add(tf.add(h1_mc_W2_mc_mc, h1_cp_W2_cp_mc), h1_ac_W2_ac_mc), self.b2_mc_mc)) # MountainCar second hidden layer
                
                h1_ac_W2_ac_ac = tf.matmul(self.h1_ac, self.W2_ac_ac)
                h1_cp_W2_cp_ac = tf.matmul(self.h1_cp, self.W2_cp_ac)
                self.h2_ac = tf.nn.relu(tf.add(tf.add(h1_ac_W2_ac_ac, self.b2_ac_ac), h1_cp_W2_cp_ac)) # Acrobot second hidden layer
                
                # Third floor weights
                self.W3_mc_mc = tf.get_variable("W3_mc_mc", [32, self.action_size], initializer=tf2_initializer)  # h2 MountainCar to h3 MountainCar weights
                trainable_list.append(self.W3_mc_mc)
                self.W3_ac_mc = tf.get_variable("W3_ac_mc", [32, self.action_size], initializer=tf2_initializer)  # h2 Acrobot to h3 MountainCar weights
                trainable_list.append(self.W3_ac_mc)
                self.b3_mc_mc = tf.get_variable("b3_mc_mc", [self.action_size]    , initializer=tf2_initializer)  # h2 to h3 bias MountainCar
                trainable_list.append(self.b3_mc_mc)
                
                # Third floor layers 
                h2_mc_W3_mc_mc = tf.matmul(self.h2_mc, self.W3_mc_mc)
                h2_ac_W3_ac_mc = tf.matmul(self.h2_ac, self.W3_ac_mc)
                self.output = tf.add(tf.add(h2_mc_W3_mc_mc, self.b3_mc_mc), h2_ac_W3_ac_mc) # Acrobot second hidden layer
                
                # extract probability distribution from output
                self.actions_distribution = tf.squeeze(self.output[:, :2])
                self.mu = tf.clip_by_value(self.actions_distribution[0], -1, 1)
                self.sigma = tf.nn.softplus(self.actions_distribution[1]) + 1e-5
                
                self.output_action = tf.clip_by_value(tf.squeeze(tf.random.normal([1], self.mu, self.sigma, tf.float32, seed=1), axis=0), -1, 1)
                self.norm_dist = tf.distributions.Normal(self.mu, self.sigma)
                
                # Loss with log probability
                self.neg_log_prob = -tf.log(self.norm_dist.prob(self.action) + 1e-5)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.I_factor * self.A_t)

            if env_name.startswith('CartPole-v1'):
                self.action = tf.placeholder(tf.float32, [self.action_size], name="action")
                trainable_list = []
                mountain_car_weights, mountain_car_layers = get_model_layers(self.state, 'MountainCarContinuous-v0', net_name)
                acrobot_weights, acrobot_layers           = get_model_layers(self.state, 'Acrobot-v1', net_name)
                
                # First floor weights
                self.W1_cp = tf.get_variable("W1_cp", [self.state_size, 64], initializer=tf2_initializer) 
                trainable_list.append(self.W1_cp)
                self.b1_cp = tf.get_variable("b1_mc", [64]                 , initializer=tf2_initializer)
                trainable_list.append(self.b1_cp)
                
                # First floor layers
                self.h1_cp = tf.nn.relu(tf.add(tf.matmul(self.state, self.W1_cp), self.b1_cp)) # Cartpole first hidden
                self.h1_mc = mountain_car_layers["h1"]  # MountainCar first hidden
                self.h1_ac = acrobot_layers["h1"]   # Acrobot first hidden
                
                # Second floor weights
                self.W2_cp_cp = tf.get_variable("W2_cp_cp", [64, self.action_size], initializer=tf2_initializer)  # h1 MountainCar to h2 MountainCar weights
                trainable_list.append(self.W2_cp_cp)
                self.W2_ac_cp = tf.get_variable("W2_ac_cp", [64, self.action_size], initializer=tf2_initializer)  # h1 Cartpole to h2 MountainCar weights
                trainable_list.append(self.W2_ac_cp)
                self.W2_mc_cp = tf.get_variable("W2_mc_cp", [64, self.action_size], initializer=tf2_initializer)  # h1 Acrobot to h2 MountainCar weights
                trainable_list.append(self.W2_mc_cp)
                self.b2_cp_cp = tf.get_variable("b2_mc_mc", [self.action_size]    , initializer=tf2_initializer)  # h1 to h2 bias MountainCar
                trainable_list.append(self.b2_cp_cp)
                
                # Second floor layers         
                h1_cp_W2_cp_cp = tf.matmul(self.h1_cp, self.W2_cp_cp)
                h1_ac_W2_ac_cp = tf.matmul(self.h1_ac, self.W2_ac_cp)
                h1_mc_W2_mc_cp = tf.matmul(self.h1_mc, self.W2_mc_cp)                
                self.output = tf.add(tf.add(tf.add(h1_cp_W2_cp_cp, h1_ac_W2_ac_cp), h1_mc_W2_mc_cp), self.b2_cp_cp) # MountainCar second hidden layer
                
                # Softmax probability distribution over actions
                self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output[:, :env_action_size]))
                
                # Loss with negative log probability
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.I_factor * self.A_t)
                
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=trainable_list)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    

# Critic
class ValueNetwork:
    def __init__(self, state_size, learning_rate, net_name='value_network', env_name = 'CartPole-v1'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(net_name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.A_t = tf.placeholder(tf.float32, name="discounted_advantage")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.I_factor = tf.placeholder(tf.float32, name="I_factor")
            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            trainable_list = []
            
            # Load individual models weights and layers
            acrobot_weights, acrobot_layers = get_model_layers(self.state, 'Acrobot-v1', net_name)
            if env_name.startswith('MountainCarContinuous-v0'):
                cartpole_weights, cartpole_layers = get_model_layers(self.state, 'CartPole-v1', net_name)
            
                ### First floor weights ###
                self.W1_mc = tf.get_variable("W1_mc", [self.state_size, 64], initializer=tf2_initializer) 
                trainable_list.append(self.W1_mc)
                
                self.b1_mc = tf.get_variable("b1_mc", [64]                 , initializer=tf2_initializer)
                trainable_list.append(self.b1_mc)
                
                ### First floor layers ###
                self.h1_mc = tf.nn.relu(tf.add(tf.matmul(self.state, self.W1_mc), self.b1_mc))  # MountainCar first hidden
                self.h1_cp = cartpole_layers["h1"]  # Cartpole first hidden
                self.h1_ac = acrobot_layers["h1"]   # Acrobot first hidden
                
                ### Second floor weights ###
                self.W2_mc_mc = tf.get_variable("W2_mc_mc", [64, 16], initializer=tf2_initializer)  # h1 MountainCar to h2 MountainCar weights
                trainable_list.append(self.W2_mc_mc)
                
                self.W2_ac_mc = tf.get_variable("W2_ac_mc", [64, 16], initializer=tf2_initializer)  # h1 Acrobot to h2 MountainCar weights
                trainable_list.append(self.W2_ac_mc)
                
                self.W2_cp_mc = tf.get_variable("W2_cp_mc", [64, 16], initializer=tf2_initializer)  # h1 Cartpole to h2 MountainCar weights
                trainable_list.append(self.W2_cp_mc)
                
                self.b2_mc_mc = tf.get_variable("b2_mc_mc", [16]    , initializer=tf2_initializer)  # h1 to h2 bias MountainCar
                trainable_list.append(self.b2_mc_mc)
                
                self.W2_cp_ac = tf.get_variable("W2_cp_ac", [64, 16], initializer=tf2_initializer)  # h1 Cartpole to h2 Acrobot weights
                trainable_list.append(self.W2_cp_ac)
                
                self.W2_ac_ac = acrobot_weights["W2"]    # h1 Acrobot to h2 Acrobot weight
                self.b2_ac_ac = acrobot_weights["b2"]    # h1 Acrobot to h2 Acrobot bias
                self.W2_cp_cp = cartpole_weights["W2"]   # h1 Cartpole to h2 Cartpole weight
                self.b2_cp_cp = cartpole_weights["b2"]   # h1 Cartpole to h2 Cartpole bias
                
                ### Second floor layers ###         
                h1_mc_W2_mc_mc = tf.matmul(self.h1_mc, self.W2_mc_mc)
                h1_cp_W2_cp_mc = tf.matmul(self.h1_cp, self.W2_cp_mc)
                h1_ac_W2_ac_mc = tf.matmul(self.h1_ac, self.W2_ac_mc)                
                self.h2_mc = tf.nn.relu(tf.add(tf.add(tf.add(h1_mc_W2_mc_mc, h1_cp_W2_cp_mc), h1_ac_W2_ac_mc), self.b2_mc_mc)) # MountainCar second hidden layer
                
                h1_ac_W2_ac_ac = tf.matmul(self.h1_ac, self.W2_ac_ac)
                h1_cp_W2_cp_ac = tf.matmul(self.h1_cp, self.W2_cp_ac)
                self.h2_ac = tf.nn.relu(tf.add(tf.add(h1_ac_W2_ac_ac, self.b2_ac_ac), h1_cp_W2_cp_ac)) # Acrobot second hidden layer
                self.h2_cp = cartpole_layers["h2"] # Cartpole second hidden layer

                ### Third floor weights ###
                self.W3_mc_mc = tf.get_variable("W3_mc_mc", [16, 1], initializer=tf2_initializer)  # h2 MountainCar to h3 MountainCar weights
                trainable_list.append(self.W3_mc_mc)
                
                self.W3_ac_mc = tf.get_variable("W3_ac_mc", [16, 1], initializer=tf2_initializer)  # h2 Acrobot to h3 MountainCar weights
                trainable_list.append(self.W3_ac_mc)
                
                self.W3_cp_mc = tf.get_variable("W3_cp_mc", [16, 1], initializer=tf2_initializer)  # h2 Cartpole to h2 MountainCar weights
                trainable_list.append(self.W3_cp_mc)
                
                self.b3_mc_mc = tf.get_variable("b3_mc_mc", [1]    , initializer=tf2_initializer)  # h1 to h2 bias MountainCar
                trainable_list.append(self.b3_mc_mc)
                
                ### Third floor layers ###         
                h2_mc_W3_mc_mc = tf.matmul(self.h2_mc, self.W3_mc_mc)
                h2_cp_W3_cp_mc = tf.matmul(self.h2_cp, self.W3_cp_mc)
                h2_ac_W3_ac_mc = tf.matmul(self.h2_ac, self.W3_ac_mc)
                                
                self.output = tf.add(tf.add(tf.add(h2_mc_W3_mc_mc, h2_cp_W3_cp_mc), h2_ac_W3_ac_mc), self.b3_mc_mc) # MountainCar output                            
            
            elif env_name.startswith('CartPole-v1'):
                mountain_car_weights, mountain_car_layers = get_model_layers(self.state, 'MountainCarContinuous-v0', net_name)  
            
                ### First floor weights ###
                self.W1_cp = tf.get_variable("W1_cp", [self.state_size, 64], initializer=tf2_initializer) 
                trainable_list.append(self.W1_cp)
                
                self.b1_cp = tf.get_variable("b1_cp", [64]                 , initializer=tf2_initializer)
                trainable_list.append(self.b1_cp)
                
                ### First floor layers ###
                self.h1_cp = tf.nn.relu(tf.add(tf.matmul(self.state, self.W1_cp), self.b1_cp))  # Cartpole first hidden
                self.h1_mc = mountain_car_layers["h1"]  # MountainCar first hidden
                self.h1_ac = acrobot_layers["h1"]   # Acrobot first hidden
                
                ### Second floor weights ###
                self.W2_cp_cp = tf.get_variable("W2_cp_cp", [64, 16], initializer=tf2_initializer)  # h1 Cartpole to h2 Cartpole weights
                trainable_list.append(self.W2_cp_cp)
                
                self.b2_cp_cp = tf.get_variable("b2_cp_cp", [16]    , initializer=tf2_initializer)  # h1 to h2 bias Cartpole
                trainable_list.append(self.b2_cp_cp)

                self.W2_ac_cp = tf.get_variable("W2_ac_cp", [64, 16], initializer=tf2_initializer)  # h1 Acrobot to h2 Cartpole weights
                trainable_list.append(self.W2_ac_cp)
                
                self.W2_mc_cp = tf.get_variable("W2_mc_cp", [64, 16], initializer=tf2_initializer)  # h1 MountainCar to h2 Cartpole weights
                trainable_list.append(self.W2_mc_cp)
                                
                self.W2_mc_ac = tf.get_variable("W2_mc_ac", [64, 16], initializer=tf2_initializer)  # h1 MountainCar to h2 Acrobot weights
                trainable_list.append(self.W2_mc_ac)
                
                self.W2_ac_ac = acrobot_weights["W2"]        # h1 Acrobot to h2 Acrobot weight
                self.b2_ac_ac = acrobot_weights["b2"]        # h1 Acrobot to h2 Acrobot bias
                self.W2_mc_mc = mountain_car_weights["W2"]   # h1 Cartpole to h2 Cartpole weight
                self.b2_mc_mc = mountain_car_weights["b2"]   # h1 Cartpole to h2 Cartpole bias
                
                ### Second floor layers ###         
                h1_cp_W2_cp_cp = tf.matmul(self.h1_cp, self.W2_cp_cp)
                h1_ac_W2_ac_cp = tf.matmul(self.h1_ac, self.W2_ac_cp)                
                h1_mc_W2_mc_cp = tf.matmul(self.h1_mc, self.W2_mc_cp)
                self.h2_cp = tf.nn.relu(tf.add(tf.add(tf.add(h1_cp_W2_cp_cp, h1_ac_W2_ac_cp), h1_mc_W2_mc_cp), self.b2_cp_cp)) # MountainCar second hidden layer
                
                h1_ac_W2_ac_ac = tf.matmul(self.h1_ac, self.W2_ac_ac)
                h1_mc_W2_mc_ac = tf.matmul(self.h1_mc, self.W2_mc_ac)
                self.h2_ac = tf.nn.relu(tf.add(tf.add(h1_ac_W2_ac_ac, self.b2_ac_ac), h1_mc_W2_mc_ac)) # Acrobot second hidden layer
                self.h2_mc = mountain_car_layers["h2"] # Cartpole second hidden layer

                ### Third floor weights ###
                self.W3_cp_cp = tf.get_variable("W3_cp_cp", [16, 1], initializer=tf2_initializer)  # h2 Cartpole to h3 Cartpole weights
                trainable_list.append(self.W3_cp_cp)
                
                self.W3_ac_cp = tf.get_variable("W3_ac_cp", [16, 1], initializer=tf2_initializer)  # h2 Acrobot to h3 Cartpole weights
                trainable_list.append(self.W3_ac_cp)
                
                self.W3_mc_cp = tf.get_variable("W3_mc_cp", [16, 1], initializer=tf2_initializer)  # h2 MountainCar to h2 Cartpole weights
                trainable_list.append(self.W3_mc_cp)
                
                self.b3_cp_cp = tf.get_variable("b3_cp_cp", [1]    , initializer=tf2_initializer)  # h2 to h3 bias Cartpole
                trainable_list.append(self.b3_cp_cp)
                
                ### Third floor layers ###         
                h2_cp_W3_cp_cp = tf.matmul(self.h2_cp, self.W3_cp_cp)
                h2_ac_W3_ac_cp = tf.matmul(self.h2_ac, self.W3_ac_cp)
                h2_mc_W3_mc_cp = tf.matmul(self.h2_mc, self.W3_mc_cp)
                                
                self.output = tf.add(tf.add(tf.add(h2_cp_W3_cp_cp, h2_ac_W3_ac_cp), h2_mc_W3_mc_cp), self.b3_cp_cp) # Cartpole output  
            
            # Mean squared error loss
            self.mse = tf.losses.mean_squared_error(predictions=self.output, labels=self.R_t)
            self.loss = tf.reduce_mean(self.mse * self.I_factor)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=trainable_list)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

def run_mountain_car(discount_factor: float, policy_learning_rate: float ,sv_learning_rate: float,
        env_name: str, desired_goal=475, action_space= [0, 1, 2], max_steps = 501):
    # Assign seed for repetitive randomness
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    
    # Initialize general settings    
    rewards, mean_rewards, losses = [], [], []
    discount_factor = discount_factor
    policy_learning_rate = policy_learning_rate
    sv_learning_rate = sv_learning_rate
    render = False
    max_episodes = 250
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
    policy = PolicyNetwork(unified_state_size, unified_action_size, env_action_size, policy_learning_rate, env_name=env_name)
    state_value = ValueNetwork(unified_state_size, sv_learning_rate, env_name=env_name)
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
                break

            rewards.append(episode_rewards[episode])
            mean_rewards.append(average_rewards)
            losses.append(loss_policy)
    return episode, rewards, mean_rewards, losses

def run_cartpole(discount_factor: float, policy_learning_rate: float ,sv_learning_rate: float,
        env_name: str, desired_goal=475, action_space= [0, 1, 2], max_steps = 501):
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
    env_action_size = env.action_space.n
    env_state_size = env.observation_space.shape[0]
    max_steps = max_steps     # defined manually

    # Initialize the policy and the state-value network
    tf.reset_default_graph()
    policy = PolicyNetwork(unified_state_size, unified_action_size, env_action_size,
                           policy_learning_rate, env_name=env_name)
    state_value = ValueNetwork(unified_state_size, sv_learning_rate, env_name=env_name)

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
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(action_space, p=actions_distribution)
                modified_action = np.array(action, dtype=np.float32, ndmin=1)
                next_state, reward, done, _ = env.step(action)

                next_state = np.pad(next_state, (0, unified_state_size - env_state_size))  # adjust state size
                next_state = next_state.reshape([1, unified_state_size])

                # encode actions
                action_one_hot = np.zeros(unified_action_size) # "EMPTY" actions encoded as 0
                action_one_hot[action] = 1
                episode_rewards[episode] += reward
                
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
                             policy.action: action_one_hot, policy.I_factor : I_factor}
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
                break
            rewards.append(episode_rewards[episode])
            mean_rewards.append(average_rewards)
            losses.append(loss_policy)
    return episode, rewards, mean_rewards, losses


if __name__ == '__main__':
    SEED = 42
    # env_name = 'MountainCarContinuous-v0'
    env_name = 'CartPole-v1'
    env_run_func = {"CartPole-v1" : run_cartpole,
                    'MountainCarContinuous-v0' : run_mountain_car}
    optimal_sv_lr = {'CartPole-v1' : 0.003, 'Acrobot-v1' : 0.0005, 'MountainCarContinuous-v0' : 0.005}
    optimal_policy_lr = {'CartPole-v1' : 0.003, 'Acrobot-v1' : 0.0005, 'MountainCarContinuous-v0' : 0.00045}
    optimal_df = {'CartPole-v1' : 0.99, 'Acrobot-v1' : 0.99, 'MountainCarContinuous-v0' : 0.99}
    env_goal      = {'CartPole-v1' : 475, 'Acrobot-v1'  : -90, 'MountainCarContinuous-v0' : 75}
    max_steps      = {'CartPole-v1' : 501, 'Acrobot-v1'  : 501, 'MountainCarContinuous-v0' : 1000}
    actions_space = {'CartPole-v1' : [0, 1], 'Acrobot-v1' : [0, 1, 2],
                     'MountainCarContinuous-v0' : np.linspace(-1, 1, 32)}
    
    # TODO: Dont forget to add 0.00001, 0.00005 to list when plotting
    # for lr in [0.00001, 0.00005, 0.00004, 0.00006, 0.00007 ,0.00009, 0.0001, 0.0002, 0.0004, 0.0005, 0.0006, 0.0007, 0.0009, 0.001, 0.003, 0.005, 0.007, 0.01]:                                                    

    for lr in [0.00001, 0.00005, 0.00004, 0.00006, 0.00007 ,0.00009,
               0.0001, 0.0002, 0.0004, 0.00045, 0.0005, 0.0006, 0.0007, 0.0009,
               0.001, 0.002, 0.003, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.007, 0.008, 0.01,
               0.02, 0.03, 0.04, 0.05, 0.07, 0.09]:                                                                                                                                                                                                                                                                                            
        last_episode, rewards, mean_rewards, losses = env_run_func[env_name](env_name=env_name,
                                                            discount_factor=optimal_df[env_name],
                                                            policy_learning_rate=0.0005,
                                                            sv_learning_rate=lr,
                                                            desired_goal=env_goal[env_name],
                                                            action_space=actions_space[env_name],
                                                            max_steps=max_steps[env_name])
        with open(os.getcwd() + '\simulations\simulationsQ3\{}\{}_sv_sim_lr={}_policy=0.0005.npy'.format(env_name, env_name, lr), 'wb') as f:
            np.save(f, last_episode)
            np.save(f, rewards)
            np.save(f, mean_rewards)
            np.save(f, losses)