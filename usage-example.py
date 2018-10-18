import gym
import gym_rock_paper_scissors

env = gym.make('RockPaperScissors-v0')

ob = env.reset()
print("Initial observation {}\n".format(ob))
for i in range(0, 3):
    fixed_action = [1, 2]
    ob, reward, done, info = env.step(fixed_action)
    print("Observation: {}\nReward: {}\nRepetition: {}\n".format(ob, reward, env.repetition))

print("Done!")
