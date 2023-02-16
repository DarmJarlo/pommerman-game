'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
from pommerman.agents.A2C.A2CAgent import A2CAgent
import numpy as np
import matplotlib.pyplot as plt
import os
global reward_end1, reward_end2

def main():
    # Create a set of agents (exactly four)
    a2c_agent_1 = A2CAgent(alpha=0.001, gamma=0.95, id=1)
    a2c_agent_2 = A2CAgent(alpha=0.001, gamma=0.95, id=2)
    agent_list = [
        a2c_agent_1,    # Left UP 1
        agents.SimpleAgent(),
        a2c_agent_2,
        agents.SimpleAgent()
    ]

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

        a2c_agent_1.board_cells = np.zeros((11, 11))
        a2c_agent_2.board_cells = np.zeros((11, 11))
        a2c_agent_1.my_bombs = np.zeros((11, 11))
        a2c_agent_2.my_bombs = np.zeros((11, 11))
        a2c_agent_1.my_bomb_life = np.zeros((11, 11))
        a2c_agent_2.my_bomb_life = np.zeros((11, 11))

        while not done:
            # env.render()
            actions = env.act(state)
            next_state, reward, done, info = env.step(actions)

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
        if len(action_history_1) < 50:  # remove the episode that enemy may suicide
            valid_episode -= 1
            continue
        reward_history_1[-1] +=  reward_end1
        reward_history_2[-1] +=  reward_end2
        sum_rewards_1 = sum(reward_history_1)
        sum_rewards_2 = sum(reward_history_2)
        ep_rewards_1.append(sum_rewards_1)
        ep_rewards_2.append(sum_rewards_2)
        avg_reward_1 = np.mean(ep_rewards_1[-100:])
        avg_reward_2 = np.mean(ep_rewards_2[-100:])
        total_avg_1.append(avg_reward_1)
        total_avg_2.append(avg_reward_2)

        print("Agent1\nGame: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}".format(i_episode, win_rate_1, sum_rewards_1, avg_reward_1, len(action_history_1), action_history_1[-1]))
        print("Agent2\nGame: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}\n".format(i_episode, win_rate_2, sum_rewards_2, avg_reward_2, len(action_history_2), action_history_2[-1]))

        """
        with open("game_log.txt", "a+") as f:
            f.write("Agent1\nGame: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}\n".format(i_episode, win_rate_1, sum_rewards_1, avg_reward_1, len(action_history_1), action_history_1[-1]))
            f.write("Agent2\nGame: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}\n\n".format(i_episode, win_rate_2, sum_rewards_2, avg_reward_2, len(action_history_2), action_history_2[-1]))
        """

        actor_loss_1, critic_loss_1 = a2c_agent_1.train_nn(state_history_1, action_history_1, reward_history_1, dones)
        actor_loss_2, critic_loss_2 = a2c_agent_2.train_nn(state_history_2, action_history_2, reward_history_2, dones)

        """
        PLOTS OF LOSSES AND WINRATE
        """
        """
        loss1.append(actor_loss_1)
        loss2.append(actor_loss_2)
        loss3.append(critic_loss_1)
        loss4.append(critic_loss_2)

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
        """

        """
        with open("losses_1.txt", "a+") as f:
            f.write(str(actor_loss_1) + " " + str(critic_loss_1) + "\n")

        with open("losses_2.txt", "a+") as f:
            f.write(str(actor_loss_2) + " " + str(critic_loss_2) + "\n")
        """

        actor_losses_1.append(actor_loss_1)
        actor_losses_2.append(actor_loss_2)
        critic_losses_1.append(critic_loss_1)
        critic_losses_2.append(critic_loss_2)

    env.close()
    #a2c_agent_1.save_model()
    #a2c_agent_2.save_model()

if __name__ == '__main__':
    main()
