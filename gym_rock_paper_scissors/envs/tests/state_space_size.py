import numpy as np
from .. import RockPaperScissorsEnv
from .. import Action


def test_one_hot_encoding():
    env = RockPaperScissorsEnv()
    assert np.array_equal(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), env.one_hot_encode_action_into_state([Action.ROCK, Action.ROCK]))
    assert np.array_equal(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), env.one_hot_encode_action_into_state([Action.ROCK, Action.PAPER]))
    assert np.array_equal(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), env.one_hot_encode_action_into_state([Action.ROCK, Action.SCISSORS]))
    assert np.array_equal(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), env.one_hot_encode_action_into_state([Action.PAPER, Action.ROCK]))
    assert np.array_equal(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), env.one_hot_encode_action_into_state([Action.PAPER, Action.PAPER]))
    assert np.array_equal(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]), env.one_hot_encode_action_into_state([Action.PAPER, Action.SCISSORS]))
    assert np.array_equal(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), env.one_hot_encode_action_into_state([Action.SCISSORS, Action.ROCK]))
    assert np.array_equal(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), env.one_hot_encode_action_into_state([Action.SCISSORS, Action.PAPER]))
    assert np.array_equal(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), env.one_hot_encode_action_into_state([Action.SCISSORS, Action.SCISSORS]))


def test_one_hot_decoding():
    env = RockPaperScissorsEnv()
    assert np.array_equal(env.decode_partial_state(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])), [Action.ROCK, Action.ROCK])
    assert np.array_equal(env.decode_partial_state(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])), [Action.ROCK, Action.PAPER])
    assert np.array_equal(env.decode_partial_state(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])), [Action.ROCK, Action.SCISSORS])
    assert np.array_equal(env.decode_partial_state(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])), [Action.PAPER, Action.ROCK])
    assert np.array_equal(env.decode_partial_state(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])), [Action.PAPER, Action.PAPER])
    assert np.array_equal(env.decode_partial_state(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])), [Action.PAPER, Action.SCISSORS])
    assert np.array_equal(env.decode_partial_state(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])), [Action.SCISSORS, Action.ROCK])
    assert np.array_equal(env.decode_partial_state(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])), [Action.SCISSORS, Action.PAPER])
    assert np.array_equal(env.decode_partial_state(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])), [Action.SCISSORS, Action.SCISSORS])


def test_calculates_correct_state_space_size_for_single_memory():
    compare_actual_state_space_size_against_expected(stacked_observations=1, expected_size=10)
    compare_actual_state_space_size_against_expected(stacked_observations=2, expected_size=91)
    compare_actual_state_space_size_against_expected(stacked_observations=3, expected_size=820)


def test_hashing_state():
    env = RockPaperScissorsEnv()
    assert 0 == env.hash_state([np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])])
    assert 0 == env.hash_state([np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) for _ in range(2)])
    assert 0 == env.hash_state([np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) for _ in range(3)])
    assert 1 == env.hash_state([np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])])
    assert 2 == env.hash_state([np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])])
    assert 3 == env.hash_state([np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])])
    assert 9 == env.hash_state([np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])])
    assert 89 == env.hash_state([np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])])
    assert 90 == env.hash_state([np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])])


def compare_actual_state_space_size_against_expected(stacked_observations, expected_size, number_of_actions=3):
    env = RockPaperScissorsEnv(stacked_observations=stacked_observations)
    actual_size = env.calculate_state_space_size(stacked_observations, number_of_actions)
    assert expected_size == actual_size
