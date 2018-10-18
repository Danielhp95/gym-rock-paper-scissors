import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='RockPaperScissors-v0',
    entry_point='gym_rock_paper_scissors.envs:RockPaperScissorsEnv',
)
