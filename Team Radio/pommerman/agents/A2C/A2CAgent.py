import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.losses as kls
from tensorflow.python.keras.backend import dtype, shape
from tensorflow.python.keras.engine import training
import tensorflow_probability as tfp
from pommerman.agents.base_agent import BaseAgent
from pommerman.agents.A2C.CombinedNN import CombinedNN
from pommerman.agents.A2C.action_prune import * #https://github.com/BorealisAI/pommerman-baseline
from pommerman.agents.A2C.message_encodings import decode_dict, encode_dict
from pommerman import utility
from pommerman import characters

class A2CAgent(BaseAgent):
    def __init__(self, alpha, gamma, id ,character=characters.Bomber):
        super(A2CAgent, self).__init__()
        self.gamma = gamma
        self.id = str(id) #remove for docker
        self.optimizer = keras.optimizers.Adam(learning_rate=alpha)

        self.n_valid_actions = 0
        self.n_invalid_actions = 0
        self._character = character
        # this variables are used for the reward
        self.board_cells = np.zeros((11, 11))
        self.my_bombs = np.zeros((11, 11))
        self.my_bomb_life = np.zeros((11, 11))
        self.last_obs = np.zeros(shape=(11, 11, 22))
        self.obs = np.zeros(shape=(11, 11, 22))
        #self.obs = obs.copy
        self.last_two_obs = [None, None] #needed for lookahead option of the action filter

    def init_NNs(self, action_space):
        self.network = CombinedNN(action_space)
        self.network.compile(self.optimizer)

    def append_last_two_obs(self, obs):
        self.last_two_obs.pop(0)
        self.last_two_obs.append(obs)

    def translate_obs(self, obs):
        #source: https://github.com/kazyka/pommerman
        #believe state idea from https://arxiv.org/pdf/2011.00424.pdf, TODO: Save last 50 steps or implement LSTM
        
        board = obs['board'].copy()
        pos  = obs['position']
        mate_id = obs['teammate'].value
        enemies = [enemy.value for enemy in self._character.enemies if enemy.value != 9]
        #enemies = [obs['enemies'][1].value for enemy in self._character.enemies if enemy.value != 9]
        #enemies = obs['enemies']
        if mate_id == 10 or mate_id == 11:
            self_agent_id = mate_id+2
        else:
            self_agent_id = mate_id-2


        enemy1, enemy2, mate = decode_dict[obs['message']]

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

        passages = np.where(board == 0, 1.0, 0.0)
        rigid_walls = np.where(board == 1, 1.0, 0.0)
        wooden_walls = np.where(board == 2, 1.0 , 0.0)
        bombs = np.where(board == 3, 1.0, 0.0)
        flames = np.where(board == 4, 1.0, 0.0)
        fog = np.where(board == 5, 1.0, 0.0)
        extra_bomb = np.where(board == 6, 1.0 , 0.0)
        incr_range = np.where(board == 7, 1.0 , 0.0)
        kick = np.where(board == 8, 1.0 , 0.0)

        bomb_life = obs['bomb_life'] #bomb on whole  board, no only its own view
        #print(bomb_life, self.id)
        bomb_blast_strength = obs['bomb_blast_strength']
        #print(bomb_blast_strength,self.id)
        bomb_direction = obs['bomb_moving_direction']
        #agent_0 is self agent
        agent_0 = np.where(board == self_agent_id, 1.0, 0.0)
        #agent_1 is partner agent

        agent_1 = np.where(board != 5, 0.0, 1.0)
        if mate_id in board:
            agent_1 = np.where(board == mate_id, 1.0, 0.0)
        else:
            if mate == 'o':
                agent_1[:6,:] = np.where(agent_1[:6,:] == 0.0, 0.0, 1.0)
            elif mate == 'u':
                agent_1[6:,:] = np.where(agent_1[6:,:] == 0.0, 0.0, 1.0)
            else:
                agent_1 = np.zeros((11,11))

        agent_2 = np.where(board != 5, 0.0, 1.0)
        if enemies[0] in board:
            agent_2 = np.where(board == enemies[0], 1.0, 0.0)
        else:
            if enemies[0] not in obs['alive']:
                agent_2 = np.zeros((11,11))
            elif enemy1 == 'q1':
                #agent_2 = np.zeros((11,11))
                agent_2[:6,6:] = np.where(agent_2[:6,6:] == 0.0, 0.0, 1.0)
            elif enemy1 == 'q2':
                #agent_2 = np.zeros((11,11))
                agent_2[:6,:6] = np.where(agent_2[:6,:6] == 0.0, 0.0, 1.0)
            elif enemy1 == 'q3':
                #agent_2 = np.zeros((11,11))
                agent_2[6:,:6] = np.where(agent_2[6:,:6] == 0.0, 0.0, 1.0)
            elif enemy1 == 'q4':
                #agent_2 = np.zeros((11,11))
                agent_2[6:,6:] = np.where(agent_2[6:,6:] == 0.0, 0.0, 1.0)

        agent_3 = np.where(board != 5, 0.0, 1.0)
        if enemies[1] in board:
            agent_3 = np.where(board == enemies[1], 1.0, 0.0)
        else:
            if enemies[1] not in obs['alive']:
                agent_3 = np.zeros((11,11))
            elif enemy2 == 'q1':
                #agent_3 = np.zeros((11,11))
                agent_3[:6,6:] = np.where(agent_3[:6,6:] == 0.0, 0.0, 1.0)
            elif enemy2 == 'q2':
                #agent_3 = np.zeros((11,11))
                agent_3[:6,:6] = np.where(agent_3[:6,:6] == 0.0, 0.0, 1.0)
            elif enemy2 == 'q3':
                #agent_3 = np.zeros((11,11))
                agent_3[6:,:6] = np.where(agent_3[6:,:6] == 0.0, 0.0, 1.0)
            elif enemy2 == 'q4':
                #agent_3 = np.zeros((11,11))
                agent_3[6:,6:] = np.where(agent_3[6:,6:] == 0.0, 0.0, 1.0)
        
        #TODO: Alive encoding, powerups in inventory, time encoiding (how to fill single values in 11x11 matrix?)

        if enemies[0] in obs['alive']:
            agent_0_alive = np.ones((11,11))
        else:
            agent_0_alive = np.zeros((11,11))
        if enemies[1] in obs['alive']:
            agent_1_alive = np.ones((11,11))
        else:
            agent_1_alive = np.zeros((11,11))
        if mate_id in obs['alive']:
            agent_2_alive = np.ones((11,11))
        else:
            agent_2_alive = np.zeros((11,11))
        
        translated_obs = np.zeros(shape=(11, 11, 22))
        
        translated_obs[:,:,0] = passages
        translated_obs[:,:,1] = rigid_walls
        translated_obs[:,:,2] = wooden_walls
        translated_obs[:,:,3] = bombs
        translated_obs[:,:,4] = flames
        translated_obs[:,:,5] = fog
        translated_obs[:,:,6] = extra_bomb
        translated_obs[:,:,7] = incr_range
        translated_obs[:,:,8] = kick
        translated_obs[:,:,9] = bomb_life
        translated_obs[:,:,10] = bomb_blast_strength #is for all the bomb on the board
        translated_obs[:,:,11] = bomb_direction
        #If an agent with the can_kick ability moves to a cell with a bomb, then the bomb is kicked in the direction from which the agent came.
        # The ensuing motion will persist until the bomb hits a wall, another agent, or the edge of the grid.
        translated_obs[:,:,12] = agent_0
        translated_obs[:,:,13] = agent_1
        translated_obs[:,:,14] = agent_2
        translated_obs[:,:,15] = agent_3
        translated_obs[:,:,16] = agent_0_alive# 11 agent
        translated_obs[:,:,17] = agent_1_alive#13 agent
        translated_obs[:,:,18] = agent_2_alive #12-agent_index
        translated_obs[:,:,19] = np.full((11,11), obs['ammo'])
        translated_obs[:,:,20] = np.full((11,11), obs['blast_strength'])
        translated_obs[:,:,21] = np.full((11,11), int(obs['can_kick']))
        #translated_obs[:,:,22] = np.full((11,11), obs['step_count']) #maybe no benefit?

        if enemies[0] in board:
            y, x = np.where(board == enemies[0])
            if y < 6 and x > 5:
                pos_enemy1 = 'q1'
            elif y < 6 and x < 6:
                pos_enemy1 = 'q2'
            elif y > 5 and x < 6:
                pos_enemy1 = 'q3'
            elif y > 5 and x > 5:
                pos_enemy1 = 'q4'
        else:
            pos_enemy1 = 'n'
        if enemies[1] in board:
            y, x = np.where(board == enemies[1])
            if y < 6 and x > 5:
                pos_enemy2 = 'q1'
            elif y < 6 and x < 6:
                pos_enemy2 = 'q2'
            elif y > 5 and x < 6:
                pos_enemy2 = 'q3'
            elif y > 5 and x > 5:
                pos_enemy2 = 'q4'
        else:
            pos_enemy2 = 'n'

        if pos[1] < 6:
            agent_pos = 'o'
        else:
            agent_pos = 'u'
    
        enc = (pos_enemy1, pos_enemy2, agent_pos)
        message = encode_dict[enc]

        return translated_obs, message

    def act(self, obs, action_space):
        self.obs = obs.copy()
        #valid_actions = [0,1,2,3,4,5]

        valid_actions = get_filtered_actions(obs, self.last_two_obs)
        self.append_last_two_obs(obs)
        obs, message = self.translate_obs(obs)   #get 11x11x22 numpy array with each channel containing different board information
        #print(obs.size)
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        obs = tf.expand_dims(obs, axis=0)   #shape must be (1, 11, 11, 22) with 1 being the batch size
        pi, value = self.network(obs)

        p_distribution = tfp.distributions.Categorical(probs=pi)
        #print('p',p_distribution)
        action = p_distribution.sample()
        #print('action',action)
        action = int(action.numpy()[0])
        '''
        #here is for reselect, the winrate is very low
        while action not in valid_actions:
            if 5 in valid_actions:
                valid_actions.remove(5)
            action = p_distribution.sample()
            action = int(action.numpy()[0])
        return (action, int(message[0]), int(message[1]))
        '''
        """
        DEBUG STUFF - prints choosen action, valid actions and percentage of valid actions
        """
        if self.n_valid_actions + self.n_invalid_actions > 0:
            try:
                with open("Actions"+self.id+".txt", "a+") as f:
                    f.write("% Valid actions: " + str((self.n_valid_actions/(self.n_valid_actions + self.n_invalid_actions))*100) + "\n")
            except:
                pass
        try:
            with open("Actions"+self.id+".txt", "a+") as f:
                f.write("Valid actions: ")
                for a in valid_actions:
                    f.write(str(a) + " ")
                f.write("Chosen action: " + str(action) + " ")
        except:
            pass

        if action in valid_actions:
            self.n_valid_actions += 1
            try:
                with open("Actions"+self.id+".txt", "a+") as f:
                    f.write("VALID\n")
            except:
                pass
            return (action, int(message[0]), int(message[1])) #Return the action choosen by the agent (network) if it is valid
        self.n_invalid_actions += 1
        try:
            with open("Actions"+self.id+".txt", "a+") as f:
                f.write("INVALID\n")
        except:
            pass
        if 5 in valid_actions:
            valid_actions.remove(5)
        action = random.choice(valid_actions)
        return (action, int(message[0]), int(message[1])) #choose random action from valid actions if agent action is not valid
    '''
    def reward_1(self):
        """
        adapted from: https://arxiv.org/pdf/2102.11762.pdf
        to calculate the reward based on the agent actions, we compare the board information
        of the current state with the previous
        """
        obs = self.translate_obs(self.obs)[0]
        reward_1 = 0
        
        # check if a power up was taken
        if self.last_obs[:,:,6][np.nonzero(obs[:,:,12]) == 1]:
            reward_1 += 0.01
        if self.last_obs[:,:,7][np.nonzero(obs[:,:,12]) == 1]:
            reward_1 += 0.01
        if self.last_obs[:,:,8][np.nonzero(obs[:,:,12]) == 1]:
            reward_1 += 0.01
        
        # first check if a bomb was placed by the agent and if it is, add it to the array for the bombs. 
        # After that we have to get the real life of the bombs in the previous Oberservation
        # to check if a bomb exploded.
        if self.last_obs[:,:,12][np.nonzero(obs[:,:,3]) == 1]:
            self.my_bombs[np.nonzero(self.last_obs[:,:,12])] = 1
            bomb_life = self.last_obs[:,:,12][np.nonzero(self.my_bombs)]
            
            #get Position of the exploding bomb & bomb blast strenght
            # mask = np.logical_and((bomb_life == 1), (obs[:,:,9] == 0)) # the result of logical_and is about true or false
            [a, b] = np.where(bomb_life==1)
            print('a',a)
            print('b',b)

            bomb_blast_str = self.last_obs[a,b,10]
            
            # to check if the bomb wasnt kicked in the last moment
            if obs[(a - bomb_blast_str),b,4] == 1 and obs[(a + bomb_blast_str),b,4] == 1 \
                and obs[a,(b - bomb_blast_str),4] == 1 and obs[a,(b + bomb_blast_str),4] == 1:
                # check if in blast radius was a wooden wall or a enemy
                for i in range((obs['blast_strenght'] + 1)):
                    if (a - i) >= 0 and (a + i) <= 10 and (b - i) >= 0 and (b + i) <= 10:
                        # to ckeck if a wooden wall was blasted
                        if obs[a, (b - i),2] == 1 or obs[(a - i), b,2] == 1 or obs[a, (b + i),2] == 1 or obs[(a + i), b,2] == 1:
                                reward += 0.01

                        # to check if enemy 1 was eleminated
                        if not utility.position_is_fog(obs[:,:,14], (a, (b - i))) and obs[a, (b - i), 14] == 1 \
                            or not utility.position_is_fog(obs[:,:,14], ((a - i), b)) and obs[(a - i), b, 14] == 1 \
                            or not utility.position_is_fog(obs[:,:,14], (a, (b + i))) and obs[a, (b + i), 14] == 1 \
                            or not utility.position_is_fog(obs[:,:,14], ((a + i), b)) and obs[(a + i), b, 14] == 1 :
                                reward += 0.50
                        
                        # to check if enemy 2 was eleminated
                        if not utility.position_is_fog(obs[:,:,15], (a, (b - i))) and obs[a, (b - i), 15] == 1 \
                            or not utility.position_is_fog(obs[:,:,15], ((a - i), b)) and obs[(a - i), b, 15] == 1 \
                            or not utility.position_is_fog(obs[:,:,15], (a, (b + i))) and obs[a, (b + i), 15] == 1 \
                            or not utility.position_is_fog(obs[:,:,15], ((a + i), b)) and obs[(a + i), b, 15] == 1 :
                                reward += 0.50
                                
                        # to check if teammate was eleminated
                        if not utility.position_is_fog(obs[:,:,13], (a, (b - i))) and obs[a, (b - i), 13] == 1 \
                            or not utility.position_is_fog(obs[:,:,13], ((a - i), b)) and obs[(a - i), b, 13] == 1 \
                            or not utility.position_is_fog(obs[:,:,13], (a, (b + i))) and obs[a, (b + i), 13] == 1 \
                            or not utility.position_is_fog(obs[:,:,13], ((a + i), b)) and obs[(a + i), b, 13] == 1 :
                                reward -= 0.50
        
        # add agent position to visited cells
        if self.board_cells[np.nonzero(obs[:,:,12])] == 0:
            self.board_cells[np.nonzero(obs[:,:,12])] = 1
            reward += 0.01
        
        self.last_obs = obs
        return reward_1
    '''
    def reward(self,actions,agent_index,state):
        """
        adapted from: https://arxiv.org/pdf/2102.11762.pdf
        to calculate the reward based on the agent actions, we compare the board information
        of the current state with the previous
        """

        #if 0 not in self.last_obs[:, :, 18]:
         #   print('tion',actions)
        now_obs = self.translate_obs(state[agent_index])[0]
        #print('alive',state[agent_index]['alive'])
        self.last_obs = self.translate_obs(self.obs)[0]# translation return obs and message
        #print('mmmm',now_obs)
        #print(now_obs.shape)
        seed = [0,0,0,0]
        reward= 0
        reward_3 = 0
        reward_4 = 0
        #find the last and now position of this agent
        last = np.argwhere(self.last_obs[:,:,12] == 1)
        e1 = np.argwhere(self.last_obs[:,:,14] == 1 )
        #print('e1',last)
        e2 = np.argwhere(self.last_obs[:,:,15] == 1 )
        now= np.argwhere(now_obs[:,:,12] == 1)
        #print('coor',now)
        #print('dist2',abs(now_coordi[0]-enemy_1_coordi[0])+abs(now_coordi[1]-enemy_1_coordi[1]))


        # check if the agent gets closer to enemy
        #if len(e1)==2 or len(e2)==2:
        if self.is_alive:
            if len(e1)!=0 and len(e2)==0 :
            #print('last', last[0,0])
                if abs(now[0,0]-e1[0,0])+abs(now[0,1]-e1[0,1]) < \
                    abs(last[0,0]-e1[0,0])+abs(last[0,1]-e1[0,1]):
                #print('hhhhhhhhhhhhh', last[0, 0])
                    reward += 2

            elif len(e2)!=0 and len(e1)==0:
                if abs(now[0, 0] - e2[0, 0]) + abs(now[0, 1] - e2[0, 1]) < \
                     abs(last[0, 0] - e2[0, 0]) + abs(last[0, 1] - e2[0, 1]):
                #print('kkkkkkkkhh', last[0, 0])
                    reward += 2
            elif len(e2)!=0 and len(e1)!=0:
                if abs(now[0, 0] - e1[0, 0]) + abs(now[0, 1] - e1[0, 1]) < \
                    abs(last[0, 0] - e1[0, 0]) + abs(last[0, 1] - e1[0, 1]) or abs(now[0, 0] - e2[0, 0]) + abs(now[0, 1] - e2[0, 1]) < \
                         abs(last[0, 0] - e2[0, 0]) + abs(last[0, 1] - e2[0, 1]):
                #print('kkpppppppppppkhh', last[0, 0])
                    reward += 2
                #print('reward_4',reward)

        # check if a power up was taken
        if self.last_obs[:, :, 6][np.nonzero(now_obs[:, :, 12])]== 1:
            reward += 5
        if self.last_obs[:, :, 7][np.nonzero(now_obs[:, :, 12])]== 1:
            reward += 5
        if self.last_obs[:, :, 8][np.nonzero(now_obs[:, :, 12])]== 1:
            reward += 5
        # add agent position to visited cells
        a=np.argwhere(now_obs[:,:,12]==1)
        #print('a',a)
        if len(a) > 0:
            if self.board_cells[a[0,0]][a[0,1]] == 0:
                self.board_cells[a[0,0]][a[0,1]] = 1
                #print('new occupy',self.board_cells)

                reward += 2
        #else:
            #print('false',obs[:,:,12])
            #print(reward)
        # first check if a bomb was placed by the agent and if it is, add it to the array for the bombs.
        # After that we have to get the real life of the bombs in the previous Oberservation
        # to check if a bomb exploded.
        #print('alive',self.obs['alive']) #here show us 10,11,12,13,
        #print('actions',actions)
        #index = agent_index + 10
        if self.is_alive:
            if actions[agent_index][0] == 5:
        #if self.last_obs[:, :, 12][np.nonzero(obs[:, :, 3]) == 1]:
                self.my_bombs[np.nonzero(now_obs[:, :, 12])] = 1
            self.my_bomb_life[np.nonzero(self.my_bombs)] = self.last_obs[:, :, 9][np.nonzero(self.my_bombs)] #to find the bomblife in last obs where my bomb is set as 1
            #print(self.my_bomb_life)
            # get Position of t+he exploding bomb & bomb blast strength

            #mask = np.logical_and((my_bomb_life == 1), (obs[:,:,9] == 0))
            #coo= np.where(mask)# the result of logical_and is about true or false
            coo_my_bomb = np.nonzero(self.my_bomb_life == 1)
            #print('coobomb',coo_my_bomb[0])
            #coo = [coor for coor in coo_my_bomb if obs[:,:,9][coor] == 0]
            #print('coo',coo)
            if len(coo_my_bomb[0]) != 0:
                #print('coobombsize',len(coo_my_bomb[0]))
                a, b = coo_my_bomb
                if now_obs[:,:,9][a[0],b[0]] == 0: #obs 9 is the bomb life
                        #print('bomb',now_obs[:,:,9])

                        #print('09999999999999900000000')
                #[a]= coo_my_bomb[0]
                #[b]= coo_my_bomb[1]# a,b is the coordinate i think a , b may be just single instead of array ,because everytime agent left one bomb only
                    #print('a', a[0])
                    #print('b', b)
                    bomb_blast_str = self.last_obs[a[0], b[0], 10]
                    #print('str',self.last_obs[:, :, 10])
                    #print('result', bomb_blast_str)
            # to check if the bomb wasnt kicked in the last moment,obs4 is flames
                        # #if around the bomb are all fire,then we think this bomb has bombed
                    left = a[0] - 1
                    right = a[0] + 1
                    up = b[0] - 1
                    down = a[0] + 1

                    if a[0]-1<0:
                        left = 0
                    if a[0]+1>10:
                        right = 10
                    if b[0]-1<0:
                        up = 0
                    if a[0]+1>10:
                        down = 10

                    if now_obs[left,b[0],1]==1:
                        left=a[0]
                    if now_obs[right,b[0],1]==1:
                        right=a[0]
                    if now_obs[a[0],up,1]==1:
                        up=b[0]
                    if now_obs[a[0],down,1]==1:
                        down=b[0]

                     # #if around the bomb are all fire,then we think this bomb has bombed
                    if now_obs[left, b[0], 4] == 1 and now_obs[right, b[0], 4] == 1 \
                        and now_obs[a[0], up, 4] == 1 and now_obs[a[0], down, 4] == 1:
                # check if in blast radius was a wooden wall or a enemy
                        for i in range(int(bomb_blast_str)):
                            left = a[0] - i
                            right = a[0] + i
                            up = b[0] - i
                            down = b[0] + i

                            if a[0] - i < 0:
                                left = 0
                            if a[0] + i > 10:
                                right = 10
                            if b[0] - i < 0:
                                up = 0
                            if b[0] + i > 10:
                                down = 10


                                # to check if a wooden wall was blasted
                                #if around the bomb are all fire,then we think this bomb has bombed
                                #here the condi may be many times counted
                            if seed[3] !=1:
                                #print('dddasdw', seed[3])
                                if self.last_obs[a[0],up,2] == 1 or self.last_obs[left, b[0],2] == 1 or self.last_obs[a[0], down,2] == 1 or self.last_obs[right, b[0],2] == 1:
                                    #condi = [now_obs[a, (b - i),2] == 1, now_obs[(a - i), b,2] == 1, now_obs[a, (b + i),2] == 1,now_obs[(a + i), b,2] == 1]
                                    #true_condi = [condition for condition in condi if condition != False]
                                    #num = len(true_condi)
                                    #if i==0:
                                     #   num = 1
                                    #print('num',num)
                                    #print('dadaaaaaaaaaaaaaaadddd')
                                    seed[3] = 1
                                    reward += 2.5
                        #agent 2,3 are enemies

                        # to check if enemy 1 was eleminated  there is overlaping here
                            if seed[0] !=15:
                                #print('dddasdw',seed[0])
                                if not utility.position_is_fog(self.last_obs[:, :, 15], (a[0], down)) and self.last_obs[a[0], down, 15] == 1 \
                                    or not utility.position_is_fog(self.last_obs[:, :, 15], (left, b[0])) and self.last_obs[left, b[0], 15] == 1 \
                                    or not utility.position_is_fog(self.last_obs[:, :, 15], (a[0], up)) and self.last_obs[a[0], up, 15] == 1 \
                                    or not utility.position_is_fog(self.last_obs[:, :, 15], (right, b[0])) and self.last_obs[
                                    right, b[0], 15] == 1:
                                    if 11 not in state[agent_index]['alive']:
                                        if 0 not in self.last_obs[:,:,17]:
                                            #print('ppppppppxxxxx')
                                            reward += 50
                                            seed[0] = 15
                                            #print('reward',reward)

                        # to check if enemy 2 was eleminated
                            if seed[1]!=0:
                                #print('sdad',seed[1])
                                if not utility.position_is_fog(self.last_obs[:, :, 14], (a[0], down)) and self.last_obs[a[0], down, 14] == 1 \
                                    or not utility.position_is_fog(self.last_obs[:, :, 14], (left, b[0])) and self.last_obs[left, b[0], 14] == 1 \
                                    or not utility.position_is_fog(self.last_obs[:, :, 14], (a[0], up)) and self.last_obs[a[0], up, 14] == 1 \
                                    or not utility.position_is_fog(self.last_obs[:, :, 14], (right, b[0])) and self.last_obs[right, b[0], 14] == 1:
                                    if 13 not in state[agent_index]['alive']:
                                        if 0 not in self.last_obs[:,:,16]:
                                            #print('xxxxxxxxxx')
                                            reward += 50
                                            seed[1] = 14
                                            #print('reward', reward)

                        # to check if teammate was eleminatedelf.
                            if seed[2] != 13:
                                #print('dddadadadasdw', seed[2])
                                if not utility.position_is_fog(self.last_obs[:, :, 13], (a[0], down)) and self.last_obs[a[0], down, 13] == 1 \
                                    or not utility.position_is_fog(self.last_obs[:, :, 13], (left, b[0])) and self.last_obs[left, b[0], 13] == 1 \
                                    or not utility.position_is_fog(self.last_obs[:, :, 13], (a[0], up)) and self.last_obs[a[0], up, 13] == 1 \
                                    or not utility.position_is_fog(self.last_obs[:, :, 13], (right, b[0])) and self.last_obs[
                                    right, b[0], 13] == 1:
                                    if 12-agent_index not in state[agent_index]['alive'] :
                                        if 0 not in self.last_obs[:,:,18]:
                                            #print('true',12-agent_index not in state[agent_index]['alive'])
                                            #print('9999999999999')
                                            reward -= 50
                                            seed[2] =13
                                            #print('reward', reward)
                        #if np.any(self.last_obs[:, :, (agent_index+12)])== 1 and np.all(self.obs[:,:,(agent_index+12)]) == 0:
                            #reward -= 0.50
        #self.board_cells = np.zeros((11,11))
        #print('nowobs',now_obs[:,:,10]) #here nowobs and lastobs  is shown right, bomb_life and bomb_strength right
        #self.last_obs = now_obs
        #print('lastobs',self.last_obs[:,:,10])
        return reward
    
    def train_nn(self, states, actions, rewards, dones):
        #https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-2of-2-b8ceb7e059db


        states = [self.translate_obs(state)[0] for state in states]

        states = np.array(states, dtype=np.float32)
        states = tf.convert_to_tensor(states, dtype=tf.float32)

        discounted_rewards = self.discount_rewards(rewards, dones)
        print(discounted_rewards)
        discounted_rewards = np.array(rewards, dtype=np.float32)
        discounted_rewards = tf.reshape(discounted_rewards, (len(discounted_rewards),))
        actions = np.array(actions, dtype=np.int32)

        with tf.GradientTape() as gt:
            pi, value = self.network(states, training=True)
            #print('value',value)
            pi_ = pi.numpy()
            np.savetxt("pi"+self.id+".txt", pi_)
            value = tf.reshape(value, (len(value),))
            #print('value', len(value))

            advantage = tf.math.subtract(discounted_rewards, value)
            
            actor_loss = self.actor_loss(pi, actions, advantage)
            critic_loss = 1/2 * kls.mean_squared_error(discounted_rewards, value)
            combined_loss = actor_loss + critic_loss

        gradients = gt.gradient(combined_loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        #print(self.network.summary())

        return actor_loss.numpy(), critic_loss.numpy()

    def actor_loss(self, pi, actions, advantage):
        #https://towardsdatascience.com/actor-critic-with-tensorflow-2-x-part-2of-2-b8ceb7e059db

        probs = []
        log_probs = []
        for pb, a in zip(pi, actions):
            distribution = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
            
            #log_prob = distribution.log_prob(a)
            prob = distribution.prob(a)

            probs.append(prob)
            #if prob < 1e-20:
             #   prob = 1e-20
            log_prob = np.log(prob)
            if log_prob < 1e-10 and log_prob >0:
                log_prob = 1e-10
            if log_prob > -1e-10 and log_prob <0:
                log_prob = -1e-10
            #print('log',log_prob)#add offset to prob, beacause small values of prob(a) can lead do nan (log(0) is not defined)
            log_probs.append(log_prob)
        
        p_loss = []
        e_loss = []
        advantage = advantage.numpy()

        for pb, a, lpb in zip(probs, advantage, log_probs):
            a = tf.constant(a)
            policy_loss = tf.math.multiply(lpb, a)
            #if policy_loss <0 and policy_loss > -1e-8:
             #   policy_loss = -1e-8
            #if policy_loss > 0 and policy_loss < 1e-8:
             #   policy_loss = 1e-8
            entropy_loss = tf.math.negative(tf.math.multiply(pb,lpb))
            #if entropy_loss <0 and entropu_loss > -1e-20:
             #   entropy_loss = -1e-20
            #if entropy_loss > 0 and entropy_loss < 1e-20:
             #   entropy_loss = 1e-20
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)

        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        loss = -p_loss - 0.001 * e_loss
        #if loss <0 and loss > -1e-20:
         #   loss = -1e-20
        #if loss > 0 and loss < 1e-20:
         #   loss = 1e-20
        return loss

    def discount_rewards(self, rewards, dones):
        discounted = 0
        discount_rewards = []
        #print('reward',rewards)
        #length = len(rewards)
        rewards.reverse()
        dones.reverse()
        for r, d in zip(rewards, dones):
            discounted = r + self.gamma * discounted * (1.0-d)
            discount_rewards.insert(0, discounted)
        #print('disco',discount_rewards[0])
        return discount_rewards

    def save_weights(self):
        self.network.save_weights("weights/agent"+self.id)

    def load_weights(self):
        self.network.load_weights("weights/agent"+self.id)

    def save_model(self):
        self.network.save("model/agent"+self.id)

    def load_model(self):
        self.network = keras.models.load_model("model/agent"+self.id)

    def save_weights_defau(self):
        self.network.save_weights("weights/agentde" + self.id)

    def load_weights_defau(self):
        self.network.load_weights("weights/agentde" + self.id)

    def save_model_defau(self):
        self.network.save("model/agent_defau"+self.id)

    def load_model_defau(self):
        self.network = keras.models.load_model("model/agent_defau" + self.id)
