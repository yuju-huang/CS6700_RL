import tensorflow as tf
import numpy as np
import os
from enum import IntEnum
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras import optimizers

from rl import Action

class DqnAgent:
    """
    DQN Agent

    The agent that explores the game and learn how to play the game by
    learning how to predict the expected long-term return, the Q value given
    a state-action pair.
    """
    Gamma = 0.8
    LearningRate = 0.01
    RandomProb = 0.3

    QNetPostfix = "/q_net.tf"
    TargetQNetPostfix = "/target_q_net.tf"

    class Mode(IntEnum):
        TRAIN = 0
        RETRAIN = 1
        PREDICT = 2

    def __init__(self, mode, location=None):
        if location is None:
            assert mode == DqnAgent.Mode.TRAIN
            self.q_net = self._build_dqn_model()
            self.target_q_net = self._build_dqn_model()
        else:
            assert mode != DqnAgent.Mode.TRAIN
            print("Load model from ", location)
            self.q_net = self.load_model(location + DqnAgent.QNetPostfix)
            self.target_q_net = self.load_model(location + DqnAgent.TargetQNetPostfix)
            if mode == DqnAgent.Mode.RETRAIN:
                self.q_net.compile(optimizer=optimizers.Adam(learning_rate=DqnAgent.LearningRate),
                                   loss='mse', run_eagerly=True)

        assert self.q_net is not None
        assert self.target_q_net is not None
        self.q_net.summary()

    @staticmethod
    def _build_dqn_model():
        """
        Builds a deep neural net which predicts the Q values for all possible
        actions given a state. The input should have the shape of the state, and
        the output should have the same shape as the action space since we want
        1 Q value per possible action.

        :return: Q network
        """
        q_net = Sequential()
        q_net.add(Dense(64, input_dim=4, activation='relu',
                        kernel_initializer='he_uniform', dtype='float64'))
        q_net.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(3, activation='linear', kernel_initializer='he_uniform'))
        q_net.compile(optimizer=optimizers.Adam(learning_rate=DqnAgent.LearningRate),
                      loss='mse', run_eagerly=True)
        return q_net

    def random_policy(self, state):
        """
        Outputs a random action

        :param state: not used
        :return: action
        """
        return Action.random()

    def collect_policy(self, state):
        """
        Similar to policy but with some randomness to encourage exploration.

        :param state: the game state
        :return: action
        """
        if np.random.random() < DqnAgent.RandomProb:
            return self.random_policy(state)
        return self.policy(state)

    def policy(self, state):
        """
        Takes a state from the game environment and returns an action that
        has the highest Q value and should be taken as the next step.

        :param state: the current game environment state
        :return: an action
        """
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state_input)
        action = np.argmax(action_q.numpy()[0], axis=0)
        return action

    def update_target_network(self):
        """
        Updates the current target_q_net with the q_net which brings all the
        training in the q_net to the target_q_net.

        :return: None
        """
        self.target_q_net.set_weights(self.q_net.get_weights())

    def train(self, batch):
        """
        Trains the underlying network with a batch of gameplay experiences to
        help it better predict the Q values.

        :param batch: a batch of gameplay experiences
        :return: training loss
        """
        state_batch, next_state_batch, action_batch, reward_batch, done_batch \
            = batch
        current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch).numpy()
        max_next_q = np.amax(next_q, axis=1)
        for i in range(state_batch.shape[0]):
            target_q_val = reward_batch[i]
            if not done_batch[i]:
                target_q_val = (1 - DqnAgent.Gamma) * target_q_val + DqnAgent.Gamma * max_next_q[i]
                #target_q_val += DqnAgent.Gamma * max_next_q[i]
            target_q[i][action_batch[i]] = target_q_val
        training_history = self.q_net.fit(x=state_batch, y=target_q, verbose=0)
        loss = training_history.history['loss']
        return loss

    def predict(self, state):
        result = self.q_net.predict(np.expand_dims(state,axis=0))
        return np.argmax(result[0], axis=0)

    def save_model(self, location):
        print("Start save model to ", location)
        tf.keras.experimental.export_saved_model(self.q_net, location + DqnAgent.QNetPostfix)
        tf.keras.experimental.export_saved_model(self.target_q_net, location + DqnAgent.TargetQNetPostfix)
        #self.q_net.save(location + DqnAgent.QNetPostfix, save_format='tf')
        #self.q_target_net.save(location + DqnAgent.TargetQNetPostfix, save_format='tf')
        print("Done save model to ", location)
  
    def load_model(self, location):
        #return tf.keras.models.load_model(location)
        return tf.keras.experimental.load_from_saved_model(location)
