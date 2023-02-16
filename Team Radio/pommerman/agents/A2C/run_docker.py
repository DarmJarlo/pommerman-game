from pommerman.agents.A2C.A2CAgent import A2CAgent
from pommerman.runner import DockerAgentRunner


class A2CDockerAgent(DockerAgentRunner):

    def __init__(self):
        self._agent = A2CAgent(alpha=0.0005, gamma=0.99)

    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)

    def episode_end(self, reward):
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()


def main():
    '''Inits and runs a Docker Agent'''
    agent = A2CDockerAgent()
    agent.run()


if __name__ == "__main__":
    main()
