from enum import Enum
import gym
from gym.spaces import Discrete, Tuple


class Action(Enum):
    ROCK     = 0
    PAPER    = 1
    SCISSORS = 2


class RockPaperScissorsEnv(gym.Env):
    '''
    Repeated game of Rock Paper scissors
    Action space: [ROCK, PAPER, SCISSORS]
    State space: previous _n_ moves by both players, where _n_ is parameterized as "stacked_observations" in the constructor.
    Reward function: -1 for losing, +1 for wining.
    '''

    def __init__(self, stacked_observations=3, max_repetitions=10):
        '''
        :param stacked_observations: Number of action pairs to be considered as part of the state
        :param max_repetitions: Number of times the game will be played
        '''
        if not isinstance(stacked_observations, int) or stacked_observations <= 0:
            raise ValueError("Parameter stacked_observations should be an integer greater than 0")

        self.action_space       = Tuple([Discrete(len(Action) - 1)]) # Substract 1 because Action.EMPTY can never be taken by an agent
        self.observation_space  = Tuple([Tuple([Discrete(len(Action)), Discrete(len(Action))]) for _ in range(stacked_observations)])
        self.state = [None for _ in range(stacked_observations)]
        self.state_space_size = self.calculate_state_space_size(stacked_observations, len([a for a in Action]))

        self.repetition = 0

    def calculate_state_space_size(self, stacked_observations, number_of_actions):
        return sum([self.calculate_number_of_states_for_fixed_memory(number_of_actions)**memory_size
                    for memory_size in range(0, stacked_observations + 1)])

    def calculate_number_of_states_for_fixed_memory(self, number_of_actions):
        """
        TODO: Change name
        """
        return number_of_actions**2

    def hash_state(self, state, number_of_actions=3):
        """
        TODO: Document
        """
        offset = self.calculate_hash_offset(state, number_of_actions)
        filetered_state = filter(lambda x: x is not None, state)
        flattened_ternary_state = [action.value for joint_action in filetered_state for action in joint_action]
        decimal_from_ternary = sum([number_of_actions**i * value for i, value in enumerate(flattened_ternary_state[::-1])])
        return decimal_from_ternary + offset

    def calculate_hash_offset(self, state, number_of_actions):
        """
        TODO: Document
        """
        number_of_empty_actions = len(list(filter(lambda x: x is None, state)))
        number_of_offsets_to_compensate = len(state) - number_of_empty_actions
        offset = sum([(number_of_actions**2)**i for i in range(0, number_of_offsets_to_compensate)])
        return offset

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
        self.state = list(map(lambda x: None, self.state))
        return self.state

    def render(self, mode='human', close=False):
        '''
        TODO: Unimplemented
        '''
        pass
