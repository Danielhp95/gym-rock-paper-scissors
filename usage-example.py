import gym
import gym_rock_paper_scissors

env = gym.make('RockPaperScissors-v0')

ob = env.reset()
print("Initial observation {}\n".format(ob))
for i in range(0, 3):
    random_action = env.action_space.sample()
    observations, reward, done, info = env.step(random_action)
    print(f"Observation player 1: {observations[0]}\nObservation player 2: {observations[1]}\nReward: {reward}\nRepetition: {env.repetition}\n")

print("Done!")
