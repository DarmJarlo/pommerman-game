import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.losses as kls
from tensorflow.python.keras.backend import dtype, shape
from tensorflow.python.keras.engine import training
import tensorflow_probability as tfp
from ..base_agent import BaseAgent
from .ActorNN import ActorNN
from .CriticNN import CriticNN
from .CombinedNN import CombinedNN
from .action_prune import * #https://github.com/BorealisAI/pommerman-baseline
import time

class A2CAgent(BaseAgent):
    def __init__(self, alpha, gamma):
        super(A2CAgent, self).__init__()
        self.gamma = gamma

        #self.actor_optimizer = keras.optimizers.Adam(learning_rate=alpha)
        #self.critic_optimizer = keras.optimizers.Adam(learning_rate=alpha)
        self.optimizer = keras.optimizers.Adam(learning_rate=alpha)

        self.n_valid_actions = 0
        self.n_invalid_actions = 0

    def init_NNs(self, action_space):
        #self.actor = ActorNN(action_space)
        #self.critic = CriticNN()
        self.network = CombinedNN(action_space)

    def translate_obs(self, obs):
        #source: https://github.com/kazyka/pommerman
        
        board = obs['board'].copy()

        """
        Passage = 0
        Rigid = 1
        Wood = 2
        Bomb = 3
        Flames = 4
        Fog = 5
        ExtraBomb = 6
        IncrRange = 7
        Kick = 8
        """
        
        agent_0 = np.where(board == 10, 1.0, 0.0)
        agent_1 = np.where(board == 11, 1.0, 0.0)
        agent_2 = np.where(board == 12, 1.0, 0.0)
        agent_3 = np.where(board == 13, 1.0, 0.0)

        rigid_walls = np.where(board == 1, 1.0, 0.0)
        wooden_walls = np.where(board == 2, 1.0 , 0.0)

        extra_bomb = np.where(board == 6, 1.0 , 0.0)
        incr_range = np.where(board == 7, 1.0 , 0.0)
        kick = np.where(board == 8, 1.0 , 0.0)

        bomb_blast_strength = obs['blast_strength']
        bomb_life = obs['bomb_life']
        bomb_moving_direction = obs['bomb_moving_direction']
        flames = np.where(board == 4, 1.0, 0.0)
        
        #obs_radius = obs_width//2
        #pos = np.asarray(obs['position'])

        # board
        #board_pad = np.pad(board, (obs_radius,obs_radius), 'constant', constant_values=1)
        #self.board_cent = board_cent = board_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]

        # bomb blast strength
        #bbs = obs['bomb_blast_strength']
        #bbs_pad = np.pad(bbs, (obs_radius,obs_radius), 'constant', constant_values=0)
        #self.bbs_cent = bbs_cent = bbs_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]

        # bomb life
        #bl = obs['bomb_life']
        #bl_pad = np.pad(bl, (obs_radius,obs_radius), 'constant', constant_values=0)
        #self.bl_cent = bl_cent = bl_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]
        translated_obs = np.zeros(shape=(11, 11, 12))

        translated_obs[:,:,0] = agent_0
        translated_obs[:,:,1] = agent_1
        translated_obs[:,:,2] = agent_2
        translated_obs[:,:,3] = agent_3
        translated_obs[:,:,4] = rigid_walls
        translated_obs[:,:,5] = wooden_walls
        translated_obs[:,:,6] = extra_bomb
        translated_obs[:,:,7] = incr_range
        translated_obs[:,:,8] = kick
        translated_obs[:,:,9] = bomb_life
        #translated_obs[:,:,10] = bomb_blast_strength
        translated_obs[:,:,10] = bomb_moving_direction
        translated_obs[:,:,11] = flames

        return translated_obs


    def act(self, obs, action_space):
        valid_actions = get_filtered_actions(obs)
        obs = self.translate_obs(obs)   #get 11x11x12 numpy array with each channel containing different board information
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)  #shape (11,11,12)
        obs = tf.expand_dims(obs, axis=0)   #shape must be (1, 11, 11, 12) with 1 being the batch size

        #pi = self.actor(obs)
        pi, value = self.network(obs)
        #pi = pi.numpy()

        p_distribution = tfp.distributions.Categorical(probs=pi)
        # print(p_distribution)
        action = p_distribution.sample()
        # print(action)
        # print(action.numpy())
        action = int(action.numpy()[0])

        if self.n_valid_actions + self.n_invalid_actions > 0:
            try:
                with open("Actions.txt", "a+") as f:
                    f.write("% Valid actions: " + str((self.n_valid_actions/(self.n_valid_actions + self.n_invalid_actions))*100) + "\n")
            except:
                pass
        try:
            with open("Actions.txt", "a+") as f:
                f.write("Valid actions: ")
                for a in valid_actions:
                    f.write(str(a) + " ")
                f.write("Chosen action: " + str(action) + " ")
        except:
            pass

        if action in valid_actions:
            self.n_valid_actions += 1
            try:
                with open("Actions.txt", "a+") as f:
                    f.write("VALID\n")
            except:
                pass
            return action
        self.n_invalid_actions += 1
        try:
            with open("Actions.txt", "a+") as f:
                f.write("INVALID\n")
        except:
            pass
        return random.choice(valid_actions)

    def train_nn(self, states, actions, rewards):
        #https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-2of-2-b8ceb7e059db

        #state = tf.convert_to_tensor([state['board'].flatten()], dtype=tf.float32)
        #next_state = tf.convert_to_tensor([next_state['board'].flatten()], dtype=tf.float32)
        states = [self.translate_obs(state) for state in states]
        states = np.array(states, dtype=np.float32)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        #states = [tf.convert_to_tensor(state, dtype=tf.float32) for state in states] #shape (11,11,12)
        #states = [tf.expand_dims(state, axis=0) for state in states]

        discounted_rewards = self.discount_rewards(rewards)
        discounted_rewards = np.array(rewards, dtype=np.float32)
        discounted_rewards = tf.reshape(discounted_rewards, (len(discounted_rewards),))
        actions = np.array(actions, dtype=np.int32)

        with tf.GradientTape() as gt1, tf.GradientTape() as gt2:
            #pi = self.actor(states, training=True)
            pi, value = self.network(states, training=True)
            pi_ = pi.numpy()
            np.savetxt("pi.txt", pi_)
            value = tf.reshape(value, (len(value),))
            
            #temporal_difference = reward + \
            #    self.gamma*next_value*(1-int(done)) - value
            temporal_difference = tf.math.subtract(discounted_rewards, value)
            
            actor_loss = self.actor_loss(pi, actions, temporal_difference)
            critic_loss = 1/2 * kls.mean_squared_error(discounted_rewards, value)
            combined_loss = actor_loss + critic_loss
        #actor_gradients = gt1.gradient(actor_loss, self.actor.trainable_variables)
        #a = [b.numpy() for b in actor_gradients]
        #critic_gradients = gt2.gradient(critic_loss, self.critic.trainable_variables)
        #c = [b.numpy() for b in critic_gradients]
        gradients = gt1.gradient(combined_loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        #self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        #self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))


        return actor_loss.numpy(), critic_loss.numpy()#, a , c

    def actor_loss(self, pi, actions, temporal_difference):
        probs = []
        log_probs = []
        for pb, a in zip(pi, actions):
            distribution = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
            
            log_prob = distribution.log_prob(a)
            prob = distribution.prob(a)
            #log_prob = tf.math.log(prob + 1e-10)
            probs.append(prob)
            log_probs.append(log_prob)
        
        p_loss = []
        e_loss = []
        temporal_difference = temporal_difference.numpy()

        for pb, t, lpb in zip(probs, temporal_difference, log_probs):
            t =  tf.constant(t)
            policy_loss = tf.math.multiply(-lpb,t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb,lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)

        #p_loss = sum(tf.stack(p_loss))
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        loss = -p_loss - 0.0001 * e_loss

        return loss

    def discount_rewards(self, rewards):
        discounted = 0
        rewards = []
        for r in rewards[::-1]:
            discounted = r + self.gamma * discounted
            rewards.insert(0, discounted)

        return rewards

    def save_model(self):
        #self.actor.save_weights("weights/actor")
        #self.critic.save_weights("weights/critic")
        self.network.save_weights("weights/critic")

    def load_model(self):
        #self.actor.load_weights("weights/actor")
        #self.critic.load_weights("weights/actor")
        self.network.load_weights("weights/critic")
