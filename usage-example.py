import gym
import gym_rock_paper_scissors

env = gym.make('RockPaperScissors-v0')

ob = env.reset()
print("Initial observation {}\n".format(ob))
for i in range(0, 10):
    random_action = env.action_space.sample()
    observations, reward, done, info = env.step(random_action)
    hashed_state = env.hash_state(observations[0])
    decoded_observations = [env.decode_state(ob) for ob in observations]
    print(f"Observation player 1: {decoded_observations[0]}\nObservation player 2: {decoded_observations[1]}\nReward: {reward}\nRepetition: {env.repetition}")
    print(f"State hash: {hashed_state}\n")

print("Done!")
