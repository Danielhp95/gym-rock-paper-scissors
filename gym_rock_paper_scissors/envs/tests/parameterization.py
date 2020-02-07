import numpy as np
from .. import RockPaperScissorsEnv


def test_default_parameterization():
    env = RockPaperScissorsEnv()
    env.reset()
    
    _, (r1, r2), _, _ = env.step([0,0])
    assert r1 == 0, r2 == 0 
    _, (r1, r2), _, _ = env.step([0,1])
    assert r1 == -1, r2 == 1
    _, (r1, r2), _, _ = env.step([0,2])
    assert r1 == 1, r2 == -1
    _, (r1, r2), _, _ = env.step([1,0])
    assert r1 == 1, r2 == -1
    _, (r1, r2), _, _ = env.step([1,1])
    assert r1 == 0, r2 == 0 
    _, (r1, r2), _, _ = env.step([1,2])
    assert r1 == -1, r2 == 1
    _, (r1, r2), _, _ = env.step([2,0])
    assert r1 == -1, r2 == 1
    _, (r1, r2), _, _ = env.step([2,1])
    assert r1 == 1, r2 == -1
    _, (r1, r2), _, _ = env.step([2,2])
    assert r1 == 0, r2 == 0 


def test_payoffs_can_be_parameterized():
    r_p, r_s, p_s = 0.5, 1.5, 2
    env = RockPaperScissorsEnv(payoff_rock_vs_paper=r_p, payoff_rock_vs_scissors=r_s, payoff_paper_vs_scissors=p_s)

    env.reset()

    _, (r1, r2), _, _ = env.step([0,0])
    assert r1 == 0, r2 == 0 
    _, (r1, r2), _, _ = env.step([0,1])
    assert r1 == r_p, r2 == -r_p
    _, (r1, r2), _, _ = env.step([0,2])
    assert r1 == r_s, r2 == -r_s
    _, (r1, r2), _, _ = env.step([1,0])
    assert r1 == -r_p, r2 == r_p
    _, (r1, r2), _, _ = env.step([1,1])
    assert r1 == 0, r2 == 0 
    _, (r1, r2), _, _ = env.step([1,2])
    assert r1 == p_s, r2 == -p_s
    _, (r1, r2), _, _ = env.step([2,0])
    assert r1 == -r_s, r2 == r_s
    _, (r1, r2), _, _ = env.step([2,1])
    assert r1 == -p_s, r2 == p_s
    _, (r1, r2), _, _ = env.step([2,2])
    assert r1 == 0, r2 == 0 
