"""
Training loop

This module trains the DQN agent by trial and error. In this module the DQN
agent will play the game episode by episode, store the gameplay experiences
and then use the saved gameplay experiences to train the underlying model.
"""
import sys
from dqn_agent import DqnAgent
from replay_buffer import ReplayBuffer
import tensorflow as tf

from environment import Environment
from env_xapian import Xapian
from rl import Action

tf.enable_eager_execution()

UPDATE_TARGET_NET_FREQ = 5
def evaluate_training_result(env, agent):
    """
    Evaluates the performance of the current DQN agent by using it to play a
    few episodes of the game and then calculates the average reward it gets.
    The higher the average reward is the better the DQN agent performs.

    :param env: the game environment
    :param agent: the DQN agent
    :return: average reward across episodes
    """
    total_reward = 0.0
#    episodes_to_play = 10
    episodes_to_play = 1
    for i in range(episodes_to_play):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            print("evaluate_training_result state=", state, ", reward=", reward, ", action=", action, ", done=", done)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    average_reward = total_reward / episodes_to_play
    return average_reward

def collect_gameplay_experiences(env, agent, buffer):
    """
    Collects gameplay experiences by playing env with the instructions
    produced by agent and stores the gameplay experiences in buffer.

    :param env: the game environment
    :param agent: the DQN agent
    :param buffer: the replay buffer
    :return: None
    """
    state = env.reset()
    done = False
    while not done:
        action = agent.collect_policy(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -1.0
        print("collect_gameplay_experiences state=", state, ", reward=", reward, ", action=", action, ", done=", done)
        buffer.store_gameplay_experience(state, next_state,
                                         reward, action, done)
        state = next_state

class FinishAgent:
    AcceptLoss = 2
    FinishThreshold = 5

    def __init__(self):
        self.loss_list = []

    def check(self, loss):
        if loss <= FinishAgent.AcceptLoss:
            self.loss_list.append(loss)
            if len(self.loss_list) == FinishAgent.FinishThreshold:
                self.loss_list = []
                return True
        else:
            self.loss_list = []

        return False

def train_model(max_episodes, out_model_path):
    """
    Trains a DQN agent to play the CartPole game by trial and error

    :return: None
    """
    agent = DqnAgent()
    buffer = ReplayBuffer()
    env = Environment(Xapian())
    finish = FinishAgent()
#    for _ in range(100):
#        collect_gameplay_experiences(env, agent, buffer)
    for episode_cnt in range(max_episodes):
        collect_gameplay_experiences(env, agent, buffer)
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = agent.train(gameplay_experience_batch)

        avg_reward = evaluate_training_result(env, agent)
        print('Episode {0}/{1} and so far the performance is {2} and '
              'loss is {3}'.format(episode_cnt, max_episodes,
                                   avg_reward, loss[0]))
        if episode_cnt % UPDATE_TARGET_NET_FREQ == 0:
            agent.update_target_network()

        if finish.check(loss[0]) == True:
            break;

    env.close()
    print("Training finished, save model to", out_model_path)
    agent.save_model(out_model_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("arg1 as output model path")
        sys.exit()

    train_model(100, sys.argv[1])
