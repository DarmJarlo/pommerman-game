'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
from pommerman.agents.A2C.A2CAgent import A2CAgent
import numpy as np
import os


def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''

    # Create a set of agents (exactly four)
    a2c_agent = A2CAgent(alpha=0.001, gamma=0.99)
    agent_list = [
        a2c_agent,
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent()
    ]
    
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    a2c_agent.init_NNs(6)
    best = np.Infinity*-1
    ep_rewards = []
    total_avg = []
    avg_reward = np.Infinity*-1
    wins = 0
    #a2c_agent.load_model()
    # Run the episodes just like OpenAI Gym
    for i_episode in range(10000):
        done = False
        state = env.reset()
        sum_rewards = 0

        reward_history = []
        state_history = []
        action_history = []

        actor_losses = []
        critic_losses = []

        if len(ep_rewards) > 100:
            ep_rewards = ep_rewards[-100:]

        while not done:
            env.render()
            actions = env.act(state)
            next_state, reward, done, info = env.step(actions)

            reward_history.append(reward[0])
            state_history.append(state[0])
            action_history.append(actions[0])
            state = next_state
            sum_rewards += reward[0]
           
            done = (reward[0] == -1) or (reward[0] == 1) or done
            if reward[0] == 1:
                wins += 1
        
        ep_rewards.append(sum_rewards)
        avg_reward = np.mean(ep_rewards[-100:])
        total_avg.append(avg_reward)
        win_rate = (wins/(i_episode+1))*100
        print("Game: {}\t Winrate: {}\t Total reward: {}\t Avg reward: {}\t Actions: {}\t Last Action: {}".format(i_episode, win_rate, sum_rewards, avg_reward, len(action_history), action_history[-1]))
        
        actor_loss, critic_loss = a2c_agent.train_nn(state_history, action_history, reward_history)

        with open("losses.txt", "a+") as f:
            f.write(str(actor_loss) + " " + str(critic_loss) + "\n")

        #a_path = os.getcwd() + "\\gradients\\actor\\" + str(i_episode)
        #c_path = os.getcwd() + "\\gradients\\critic\\" + str(i_episode)
        #np.save(a_path, a_grad)
        #np.save(c_path, c_grad)
        #reward_history.append(sum_rewards)
        #avg = np.mean(np.array(reward_history[-100:]))

        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
    env.close()
    a2c_agent.save_model()


if __name__ == '__main__':
    main()
