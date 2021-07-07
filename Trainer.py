import tensorflow as tf
import numpy as np
import random
import time

from Source2 import BEAM
from model import DQN

tf.app.flags.DEFINE_boolean("train", False, "학습모드. 게임을 화면에 보여주지 않습니다.") 

FLAGS = tf.app.flags.FLAGS

MAX_EPISODE = 10000
TARGET_UPDATE_INTERVAL = 1000
TRAIN_INTERVAL = 4
OBSERVE = 100
sample_signals = 3

NUM_ACTION = 3
NOISE_POWER = 0.01

sess = tf.Session()

beam = BEAM(sample_signals)

brain = DQN(sess, sample_signals, NUM_ACTION)

rewards = tf.placeholder(tf.float32, [None])

tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('logs', sess.graph)

summary_merged = tf.summary.merge_all()

brain.update_target_network()

epsilon = 1.0
# 프레임 횟수
time_step = 0
total_reward_list = []

for episode in range(MAX_EPISODE):
    
    terminal = False
    total_reward = 0
    
    state = beam.reset()
    brain.init_state(state)
    
    while not terminal:
        
        if np.random.rand() < epsilon:
            action = random.randrange(NUM_ACTION)
        else:
            action = brain.get_action()
            
        if epsilon > OBSERVE:
            epsilon -= 0.001

        state, reward, terminal = beam.step(action)
        total_reward += reward
        
        brain.remember(state, action, reward, terminal)
        
        if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
            # DQN 으로 학습을 진행합니다.
            brain.train()

        if time_step % TARGET_UPDATE_INTERVAL == 0:
            # 타겟 네트웍을 업데이트 해 줍니다.
            brain.update_target_network()
            
        time_step += 1

        total_reward_list.append(total_reward)
        
    print("episode: %d reward: %f"%(time_step, total_reward)) 
    
    total_reward_list.append(total_reward)
    
    if episode % 10000 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

    if episode % 100000 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step=time_step)