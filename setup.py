from setuptools import setup, find_packages

setup(name='gym_rock_paper_scissors',
      version='0.0.125',
      description='OpenAI gym environment for a repeated game of Rock-Paper-Scissors',
      url='https://github.com/Danielhp95/gym-rock-paper-scissors',
      author='Sarios',
      author_email='rockpapersass@xcape.com',
      packages=find_packages(),
      install_requires=['gym']
      )
