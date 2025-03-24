from unityagents import UnityEnvironment


def setup_reacher_environment():

    print('\n>>>>>>>>>>>>>>> Setting up environment <<<<<<<<<<<<<<<\n\n')
    env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)

    # size of each action
    action_size = brain.vector_action_space_size

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]

    return env, brain_name, num_agents, action_size, state_size

class UnityMLEnvironemtAdapter:

    def __init__(self, env, brain_name, agent_count):

        assert(agent_count==1)

        self.env = env
        self.brain_name = brain_name
        self.brain = self.env.brains[brain_name]
        self.agent_count = agent_count
        self.state_current = None
        self.state_next = None

    def reset(self):

        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.state_current = env_info.vector_observations[0]

        return self.state_current

    def getState(self):# -> None | Any:

        return self.state_current
    
    def step(self, action):

        env_info = self.env.step(action)[self.brain_name]

        return env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]


