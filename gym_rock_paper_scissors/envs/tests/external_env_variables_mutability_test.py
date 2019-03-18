from .. import RockPaperScissorsEnv

import numpy as np


def test_changing_state_returned_by_step_does_not_change_internal_environment_state():
    env = RockPaperScissorsEnv()
    observation_1, observation_2 = env.reset()
    assert env.state[0][0] != -4
    observation_1[0] = -4
    assert env.state[0][0] != -4
    np.testing.assert_array_equal(env.state, observation_2)

    joint_action = [1, 2]
    (new_observation_1, new_observation_2), reward, done, info = env.step(joint_action)
    assert env.state[0][0] != -4
    new_observation_1[0] = -4
    assert env.state[0][0] != -4
    np.testing.assert_array_equal(env.state, new_observation_2)
