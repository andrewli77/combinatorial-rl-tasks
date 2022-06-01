=======================================================
Installation:
=======================================================

Please install the following packages before running the code. We recommend using Python3.6. 

- MuJoCo (https://mujoco.org/)
- OpenAI Safety Gym (https://github.com/openai/safety-gym).
    - The install location should be the same directory as this README file. 
- A list of versions for major dependencies is in `requirements.txt`


=======================================================
Training and Evaluating RL agents:
=======================================================

The code is separated into folders for different baselines. Instructions for training and evaluating each baseline can be found in that folder.

- To run PPO, PPO-VD, Fixed-length skills, DIAYN, or TSP-solver go to the `main` folder. 
- To run Zone-goals, go to the `zone-goals` folder.
- To run Options, go to the `options` folder. 
- To run xy-goals, go to the `xy-goals` folder.