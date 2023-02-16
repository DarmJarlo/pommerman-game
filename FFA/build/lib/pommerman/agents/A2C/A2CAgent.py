import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.backend import dtype
import tensorflow_probability as tfp
from ..base_agent import BaseAgent
from .ActorNN import ActorNN
from .CriticNN import CriticNN


class A2CAgent(BaseAgent):
    def __init__(self, alpha, gamma):
        super(A2CAgent, self).__init__()
        self.gamma = gamma

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=alpha)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=alpha)

    def init_NNs(self, n_actions):
        self.actor = ActorNN(n_actions)
        self.critic = CriticNN()

    def act(self, obs, action_space):
        # TODO: create own state representation (e.g, 11x11x3)
        p = self.actor(np.array(obs['board']))
        # print(p)
        p = p.numpy()
        # print(p)
        p_distribution = tfp.distributions.Categorical(
            probs=p, dtype=tf.float32)
        # print(p_distribution)
        action = p_distribution.sample()
        # print(action)
        # print(action.numpy())

        return int(action.numpy()[0])

    def train_nn(self, state, action, reward, next_state, done):
        state = np.array(state['board'])
        next_state = np.array(next_state['board'])

        with tf.GradientTape() as gt1, tf.GradientTape() as gt2:
            pi = self.actor(state)
            value = self.critic(state)
            next_value = self.critic(next_state)
            temporal_difference = reward + \
                self.gamma*next_value*(1-int(done)) - value
            actor_loss = self.actor_loss(pi, action, temporal_difference)
            critic_loss = temporal_difference**2

            actor_gradients = gt1.gradient(
                actor_loss, self.actor.trainable_variables)
            critic_gradients = gt2.gradient(
                critic_loss, self.critic.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_gradients, self.actor.trainable_variables))
            self.actor_optimizer.apply_gradients(
                zip(critic_gradients, self.critic.trainable_variables))

        return actor_loss, critic_loss

    def actor_loss(self, p, action, temporal_difference):
        distribution = tfp.distributions.Categorical(probs=p, dtype=tf.float32)
        log_p = distribution.log_prob(action)
        # - because of gradient ascent, TODO: Maybe change to gradient descent
        loss = -log_p*temporal_difference

        return loss

    def save_model(self):
        self.actor.save_weights("weights/actor")
        self.critic.save_weights("weights/critic")

    def load_model(self):
        self.actor.load_weights("weights/actor")
        self.critic.load_weights("weights/critic")
