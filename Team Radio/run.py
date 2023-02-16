'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
from pommerman.agents.A2C.A2CAgent import A2CAgent
import numpy as np
import matplotlib.pyplot as plt
import os
global reward_end1, reward_end2

def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''

    # Create a set of agents (exactly four)
    #a2c_agent_1 = agents.DockerAgent("tud22-group-ac.1", 10080)
    #a2c_agent_2 = agents.DockerAgent("tud22-group-ac.2", 10080)
    a2c_agent_1 = A2CAgent(alpha=0.001, gamma=0.95, id=1)
    a2c_agent_2 = A2CAgent(alpha=0.001, gamma=0.95, id=2)
    agent_list = [
        a2c_agent_1,    # Left UP 1
        #agents.RandomAgent(),
        agents.SimpleAgent(),
        #agents.RandomAgent(),# Left down 2
        a2c_agent_2,
        agents.SimpleAgent()
        #a2c_agent_1          # Right down 3 right up 4
        #agents.RandomAgent()
    ]
    #print('obs',obs['alive'])
    #print('position', a2c_agent_2.obs[:,:,14])
    agent_index_1 = agent_list.index(a2c_agent_1)
    agent_index_2 = agent_list.index(a2c_agent_2)
    enemy = [enemy_index for enemy_index in np.arange(4) if enemy_index != agent_index_2 and enemy_index != agent_index_1]
    enemy = [agent_list[enemy[0]],agent_list[enemy[1]]]
    a2c_agent_1.init_NNs(6)
    a2c_agent_2.init_NNs(6)
    #a2c_agent_1.load_model()
    #a2c_agent_2.load_model()
    # Make the "Radio" environment using the agent list
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    best = np.Infinity*-1
    ep_rewards_1 = []
    ep_rewards_2 = []
    total_avg_1 = []
    total_avg_2 = []
    avg_reward_1 = np.Infinity*-1
    avg_reward_2 = np.Infinity*-1
    wins_1 = []
    wins_2 = []
    x = []
    big_x = []
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    reward_end1 = 0
    reward_end2 = 0
    valid_episode = 0
    win_rate = []
    #a2c_agent_1.load_model()
    #a2c_agent_2.load_model()
    # Run the episodes just like OpenAI Gym
    for i_episode in range(1000):
        done = False
        state = env.reset()
        valid_episode +=1
        sum_rewards_1 = 0
        sum_rewards_2 = 0

        reward_history_1 = []
        reward_history_2 = []
        state_history_1 = []
        state_history_2 = []
        action_history_1 = []
        action_history_2 = []


        actor_losses_1 = []
        actor_losses_2 = []
        critic_losses_1 = []
        critic_losses_2 = []
        dones = []
        #reward_1 = 0
        #reward_2 = 0
        a2c_agent_1.board_cells = np.zeros((11, 11))
        a2c_agent_2.board_cells = np.zeros((11, 11))
        a2c_agent_1.my_bombs = np.zeros((11, 11))
        a2c_agent_2.my_bombs = np.zeros((11, 11))
        a2c_agent_1.my_bomb_life = np.zeros((11, 11))
        a2c_agent_2.my_bomb_life = np.zeros((11, 11))

        while not done:
            # env.render()
            actions = env.act(state)
            # in this env.act we have updated the obs
            next_state, reward, done, info = env.step(actions)
            # original reward is in forward_model FFA/629 ZEILE
            #print(next_state.size) next state is a list has no size .
            '''if reward[0] != 0:
                reward_1 = reward[0]
            else:
                reward_1 = a2c_agent_1.reward(agent_index_1)
            if reward[2] != 0:
                reward_2 = reward[2]
            else:
                reward_2 = a2c_agent_2.reward(agent_index_2)
            '''
            #print('position', a2c_agent_2.obs[:, :, 14])
            #reward_1 = a2c_agent_1.reward(actions,agent_index_1)
            #reward_2 = a2c_agent_2.reward(actions,agent_index_2)
            #reward_history_1 = a2c_agent_1.discount_rewards(reward_1)
            #reward_history_2 = a2c_agent_2.discount_rewards(reward_2)
            state_history_1.append(state[0])
            state_history_2.append(state[2])

            if type(actions[0]) == tuple:
                action_history_1.append(actions[0][0])
            else:
                action_history_1.append(actions[0])
            if type(actions[2]) == tuple:
                action_history_2.append(actions[2][0])
            else:
                action_history_2.append(actions[2])

            state = next_state

            reward_1 = a2c_agent_1.reward(actions, agent_index_1,state)
            reward_2 = a2c_agent_2.reward(actions, agent_index_2,state)
            reward_history_1.append(reward_1)
            reward_history_2.append(reward_2)

            #done = ((reward_1 == -1) or (reward_1 == 1)) and ((reward_2 == -1) or (reward_2 == 1)) or done
            if (a2c_agent_1.is_alive or a2c_agent_2.is_alive) and not (enemy[0].is_alive or enemy[1].is_alive):
                wins_1.append(1)
                reward_end1 = 1000
                wins_2.append(1)
                reward_end2 = 1000
            if ((enemy[0].is_alive or enemy[1].is_alive) and not (a2c_agent_1.is_alive or a2c_agent_2.is_alive)) or \
                not (a2c_agent_1.is_alive or a2c_agent_2.is_alive or enemy[0].is_alive or enemy[1].is_alive):
                wins_1.append(0)
                wins_2.append(0)
                reward_end1 = -1000
                reward_end2 = -1000

            #sum_rewards_1 += reward_1
            # print(sum_rewards_1)
            #sum_rewards_2 += reward_2
            #reward_history_1.append(reward_1)
            #reward_history_2.append(reward_2)
            done = done or ((a2c_agent_1.is_alive or a2c_agent_2.is_alive) and not (enemy[0].is_alive or enemy[1].is_alive)) or \
                   ((enemy[0].is_alive or enemy[1].is_alive) and not (a2c_agent_1.is_alive or a2c_agent_2.is_alive)) or \
                   not (a2c_agent_1.is_alive or a2c_agent_2.is_alive or enemy[0].is_alive or enemy[1].is_alive)
            dones.append(done)
        if len(wins_2) < 100:

            win_rate_1 = (sum(wins_1) / (i_episode + 1)) * 100
            win_rate_2 = (sum(wins_2) / (i_episode + 1)) * 100
        else:
            win_rate_1 = (sum(wins_1[-100:])/100) * 100
            win_rate_2 = (sum(wins_2[-100:]) /100) *100

        win_rate.append(win_rate_1)
        big_x.append(i_episode+1)
        if len(action_history_1) < 50:# remove the episode that enemy may suicide
            valid_episode -= 1
            continue
        reward_history_1[-1] +=  reward_end1
        reward_history_2[-1] +=  reward_end2
        sum_rewards_1 = sum(reward_history_1)
        sum_rewards_2 = sum(reward_history_2)
        ep_rewards_1.append(sum_rewards_1) #ep is epoche
        ep_rewards_2.append(sum_rewards_2)
        avg_reward_1 = np.mean(ep_rewards_1[-100:])
        avg_reward_2 = np.mean(ep_rewards_2[-100:])
        total_avg_1.append(avg_reward_1)
        total_avg_2.append(avg_reward_2)

        print("Agent1\nGame: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}".format(i_episode, win_rate_1, sum_rewards_1, avg_reward_1, len(action_history_1), action_history_1[-1]))
        print("Agent2\nGame: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}\n".format(i_episode, win_rate_2, sum_rewards_2, avg_reward_2, len(action_history_2), action_history_2[-1]))

        with open("game_log.txt", "a+") as f:
            f.write("Agent1\nGame: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}\n".format(i_episode, win_rate_1, sum_rewards_1, avg_reward_1, len(action_history_1), action_history_1[-1]))
            f.write("Agent2\nGame: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}\n\n".format(i_episode, win_rate_2, sum_rewards_2, avg_reward_2, len(action_history_2), action_history_2[-1]))

        actor_loss_1, critic_loss_1 = a2c_agent_1.train_nn(state_history_1, action_history_1, reward_history_1, dones)
        actor_loss_2, critic_loss_2 = a2c_agent_2.train_nn(state_history_2, action_history_2, reward_history_2, dones)
        loss1.append(actor_loss_1)
        loss2.append(actor_loss_2)
        loss3.append(critic_loss_1)
        loss4.append(critic_loss_2)
        #x = np.linspace(1,i_episode+1,num=i_episode,endpoint='True')
        x.append(valid_episode)

        print('x', x)
        print('y', loss2)
        print('z', loss4)
        plt.plot(x, loss1, 'bo-')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('./loss1.jpg')
        # plt.show
        plt.clf()
        plt.plot(x, loss2, 'bo-')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('./loss2.jpg')
        # plt.show
        plt.clf()
        plt.plot(x, loss3, 'bo-')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('./loss3.jpg')
        # plt.show
        plt.clf()
        plt.plot(x, loss4, 'bo-')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('./loss4.jpg')
        # plt.show
        plt.clf()
        plt.plot(big_x,win_rate , 'bo-')
        plt.xlabel('iteration')
        plt.ylabel('win_rate/%')
        plt.savefig('./winrate.jpg')
        plt.clf()
        with open("losses_1.txt", "a+") as f:
            f.write(str(actor_loss_1) + " " + str(critic_loss_1) + "\n")

        with open("losses_2.txt", "a+") as f:
            f.write(str(actor_loss_2) + " " + str(critic_loss_2) + "\n")


        actor_losses_1.append(actor_loss_1)
        actor_losses_2.append(actor_loss_2)
        critic_losses_1.append(critic_loss_1)
        critic_losses_2.append(critic_loss_2)

        #test123
    env.close()
    a2c_agent_1.save_model()
    a2c_agent_2.save_model()

def main_default_reward():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''

    # Create a set of agents (exactly four)
    #a2c_agent_1 = agents.DockerAgent("tud22-group-ac.1", 10080)
    #a2c_agent_2 = agents.DockerAgent("tud22-group-ac.2", 10080)
    a2c_agent_1 = A2CAgent(alpha=0.001, gamma=0.95, id=1)
    a2c_agent_2 = A2CAgent(alpha=0.001, gamma=0.95, id=2)
    agent_list = [
        a2c_agent_1,    # Left UP 1
        #agents.RandomAgent(),
        agents.SimpleAgent(),
        #agents.RandomAgent(),# Left down 2
        a2c_agent_2,
        agents.SimpleAgent()
        #a2c_agent_1          # Right down 3 right up 4
        #agents.RandomAgent()
    ]
    #print('obs',obs['alive'])
    #print('position', a2c_agent_2.obs[:,:,14])

    #enemy = [enemy_index for enemy_index in np.arange(4) if enemy_index != agent_index_2 and enemy_index != agent_index_1]
    #enemy = [agent_list[enemy[0]],agent_list[enemy[1]]]
    a2c_agent_1.init_NNs(6)
    a2c_agent_2.init_NNs(6)
    #a2c_agent_1.load_model()
    #a2c_agent_2.load_model()
    # Make the "Radio" environment using the agent list
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    best = np.Infinity*-1
    ep_rewards_1 = []
    ep_rewards_2 = []
    total_avg_1 = []
    total_avg_2 = []
    avg_reward_1 = np.Infinity*-1
    avg_reward_2 = np.Infinity*-1
    wins_1 = 0
    wins_2 = 0
    x = []
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    reward_end1 = 0
    reward_end2 = 0
    #a2c_agent.load_model()
    # Run the episodes just like OpenAI Gym
    for i_episode in range(1000):
        done = False
        state = env.reset()
        sum_rewards_1 = 0
        sum_rewards_2 = 0
        reward_1 = []
        reward_2 = []
        reward_history_1 = []
        reward_history_2 = []
        state_history_1 = []
        state_history_2 = []
        action_history_1 = []
        action_history_2 = []


        actor_losses_1 = []
        actor_losses_2 = []
        critic_losses_1 = []
        critic_losses_2 = []
        dones = []

        while not done:
            env.render()
            actions = env.act(state)
            next_state, reward, done, info = env.step(actions)# step in v0.py

            if type(actions[0]) == tuple:
                action_history_1.append(actions[0][0])
            else:
                action_history_1.append(actions[0])
            if type(actions[2]) == tuple:
                action_history_2.append(actions[2][0])
            else:
                action_history_2.append(actions[2])



            reward_history_1.append(reward[0])
            state_history_1.append(state[0])

            sum_rewards_1 += reward[0]

            reward_history_2.append(reward[2])
            state_history_2.append(state[2])

            state = next_state
            sum_rewards_2 += reward[2]

            done = (reward[0] == -1 and reward[2] == -1) or (reward[0] == 1 and reward[2] == 1)
            if reward[0]==1:
                wins_1 += 1
                wins_2 += 1
            dones.append(done)
            reward_1.append(reward[0])
            reward_2.append(reward[2])
            ep_rewards_1.append(reward[0])
            ep_rewards_2.append(reward[2])
        if len(ep_rewards_1) > 100:
            avg_reward_1 = np.mean(ep_rewards_1[-100:])
            avg_reward_2 = np.mean(ep_rewards_2[-100:])
        else:
            avg_reward_1 = np.mean(ep_rewards_1)
            avg_reward_2 = np.mean(ep_rewards_1)



        actor_loss_1, critic_loss_1 = a2c_agent_1.train_nn(state_history_1, action_history_1, reward_1,dones)
        actor_loss_2, critic_loss_2 = a2c_agent_2.train_nn(state_history_2, action_history_2, reward_2,dones)
            # here should i add a load model and save model
            # reward_history.append(sum_rewards)
            # avg = np.mean(np.array(reward_history[-100:]))

        if avg_reward_1 > best:
            best = avg_reward_1
            a2c_agent_1.save_model_defau()
            a2c_agent_2.save_model_defau()
            #print('position', a2c_agent_2.obs[:, :, 14])
            #reward_1 = a2c_agent_1.reward(actions,agent_index_1)
            #reward_2 = a2c_agent_2.reward(actions,agent_index_2)
            #reward_history_1 = a2c_agent_1.disco
            #done = ((reward_1 == -1) or (reward_1 == 1)) and ((reward_2 == -1) or (reward_2 == 1)) or done
        sum = np.sum(ep_rewards_1)
        total_avg_1.append(avg_reward_1)
        total_avg_2.append(avg_reward_2)
        win_rate_1 = (wins_1/(i_episode+1))*100
        win_rate_2 = (wins_2/(i_episode+1))*100
        print("Agent1\nGame: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}".format(i_episode, win_rate_1, sum, avg_reward_1, len(action_history_1), action_history_1[-1]))
        print("Agent2\nGame: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}\n".format(i_episode, win_rate_2, sum, avg_reward_2, len(action_history_2), action_history_2[-1]))

        with open("game_log_default.txt", "a+") as f:
            f.write("Agent1\nGame: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}\n".format(i_episode, win_rate_1, sum_rewards_1, avg_reward_1, len(action_history_1), action_history_1[-1]))
            f.write("Agent2\nGame: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}\n\n".format(i_episode, win_rate_2, sum_rewards_2, avg_reward_2, len(action_history_2), action_history_2[-1]))

        actor_loss_1, critic_loss_1 = a2c_agent_1.train_nn(state_history_1, action_history_1, reward_history_1, dones)
        actor_loss_2, critic_loss_2 = a2c_agent_2.train_nn(state_history_2, action_history_2, reward_history_2, dones)
        loss1.append(actor_loss_1)
        loss2.append(np.log(actor_loss_2))
        loss3.append(critic_loss_1)
        loss4.append(np.log(critic_loss_2))
        #x = np.linspace(1,i_episode+1,num=i_episode,endpoint='True')
        x.append(i_episode+1)
        print('x', x)
        print('y', loss3)

        plt.plot(x, loss1, 'bo-')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('./loss1default,jpg')
        # plt.show
        plt.clf()
        plt.plot(x, loss2, 'bo-')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('./loss2default.jpg')
        # plt.show
        plt.clf()
        plt.plot(x, loss3, 'bo-')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('./loss3default.jpg')
        # plt.show
        plt.clf()
        plt.plot(x, loss4, 'bo-')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('./loss4default.jpg')
        # plt.show
        plt.clf()

        with open("lossesdefault_1.txt", "a+") as f:
            f.write(str(actor_loss_1) + " " + str(critic_loss_1) + "\n")

        with open("lossesdefault_2.txt", "a+") as f:
            f.write(str(actor_loss_2) + " " + str(critic_loss_2) + "\n")


        actor_losses_1.append(actor_loss_1)
        actor_losses_2.append(actor_loss_2)
        critic_losses_1.append(critic_loss_1)
        critic_losses_2.append(critic_loss_2)

        #test123
    env.close()
    a2c_agent_1.save_model_defau()
    a2c_agent_2.save_model_defau()


if __name__ == '__main__':
    main()
    #main_default_reward()
