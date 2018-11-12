from .. import RockPaperScissorsEnv
from .. import Action


def test_calculates_correct_state_space_size_for_single_memory():
    compare_actual_state_space_size_against_expected(stacked_observations=1, expected_size=10)
    compare_actual_state_space_size_against_expected(stacked_observations=2, expected_size=91)
    compare_actual_state_space_size_against_expected(stacked_observations=3, expected_size=820)


def test_hashing_state():
    env = RockPaperScissorsEnv()
    assert 0 == env.hash_state([None])
    assert 0 == env.hash_state([None, None])
    assert 0 == env.hash_state([None, None, None])
    assert 1 == env.hash_state([(Action.ROCK, Action.ROCK)])
    assert 2 == env.hash_state([(Action.ROCK, Action.PAPER)])
    assert 3 == env.hash_state([(Action.ROCK, Action.SCISSORS)])
    assert 9 == env.hash_state([(Action.SCISSORS, Action.SCISSORS)])
    assert 89 == env.hash_state([(Action.SCISSORS, Action.SCISSORS), (Action.SCISSORS, Action.PAPER)])
    assert 90 == env.hash_state([(Action.SCISSORS, Action.SCISSORS), (Action.SCISSORS, Action.SCISSORS)])


def compare_actual_state_space_size_against_expected(stacked_observations, expected_size, number_of_actions=3):
    env = RockPaperScissorsEnv(stacked_observations=stacked_observations)
    actual_size = env.calculate_state_space_size(stacked_observations, number_of_actions)
    assert expected_size == actual_size
