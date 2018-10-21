from enum import Enum
import gym


class Action(Enum):
    EMPTY    = 0
    ROCK     = 1
    PAPER    = 2
    SCISSORS = 3


class RockPaperScissorsEnv(gym.Env):
    '''
    Repeated game of Rock Paper scissors
    Action space: [EMPTY, ROCK, PAPER, SCISSORS]
    State space: previous _n_ moves by both players, where _n_ is parameterized as "stacked_observations" in the constructor.
    Reward function: -1 for losing, +1 for wining.
    '''

    def __init__(self, stacked_observations=3, max_repetitions=10):
        '''
        :param stacked_observations: Number of action pairs to be considered as part of the state
        '''
        if not isinstance(stacked_observations, int) or stacked_observations <= 0:
            raise ValueError("Parameter stacked_observations should be an integer greater than 0")

        self.state = [[Action.EMPTY, Action.EMPTY] for _ in range(stacked_observations)]
        self.repetition = 0

    def step(self, action):
        '''
        Performs a step of the reinforcement learning loop by executing the action, changing the environment's state,
        computing the reward for all agents, and detecting if the environment has reached a terminal state
        :param action: vector containing an action for both players
        '''
        if not isinstance(action, list) or len(action) != 2:
            raise ValueError("Parameter action should be a vector of length 2 containing an Action for each player")
        if any(map(lambda a: a not in range(1, 4), action)):
            raise ValueError("Both actions in the action vector should be either (1) Rock, (2) Paper, (3) Scissors")

        encoded_action = [Action(a) for a in action]
        new_state        = self.transition_probability_function(self.state, encoded_action)
        reward           = self.reward_function(encoded_action)
        self.repetition += 1
        info = {}
        return new_state, reward, self.repetition == self.max_repetitions, info

    def transition_probability_function(self, current_state, action):
        '''
        Executes the :param: action in the :param: current_state, creating a new state
        :param current_state: state of the environment before action is executed
        :param action: vector containing an action for both players
        '''
        current_state.pop(0)
        current_state.append(action)
        return current_state

    def reward_function(self, action):
        '''
        Reward function for the zero sum two player game of rock paper scissors.
        Rock beats scissor, scissors beat paper, paper beats rock. If both player
        take the same action, they both get zero reward.
        :param action: action vector containing action for both players
        '''
        if action[0] == Action.ROCK and action[1] == Action.PAPER: return [-1, 1]
        if action[0] == Action.ROCK and action[1] == Action.SCISSORS: return [1, -1]
        if action[0] == Action.PAPER and action[1] == Action.ROCK: return [1, -1]
        if action[0] == Action.PAPER and action[1] == Action.SCISSORS: return [-1, 1]
        if action[0] == Action.SCISSORS and action[1] == Action.ROCK: return [-1, 1]
        if action[0] == Action.SCISSORS and action[1] == Action.PAPER: return [1, -1]
        if action[0] == action[1]: return [0, 0]
        raise ValueError("One of the player actions was empty")

    def reset(self):
        '''
        Resets state by emptying the state vector
        '''
        self.repetition = 0
        self.state = list(map(lambda x: [Action.EMPTY, Action.EMPTY], self.state))
        return self.state

    def render(self, mode='human', close=False):
        '''
        TODO: Unimplemented
        '''
        pass
