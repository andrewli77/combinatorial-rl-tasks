# [Combinatorial Optimization Tasks in MuJoCo](https://arxiv.org/pdf/2206.01812.pdf)

Most existing benchmarks for long-horizon RL involve **simple high-level task structure**, such as reaching a goal location, opening a drawer, or making a robot move as fast as possible. Why is this? Unfortunately, as tasks grow in complexity, rewards often become too sparse for RL methods that learn from scratch.

 
PointTSP | TimedTSP | ColourMatch
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/andrewli77/combinatorial-rl-tasks/blob/master/gifs/pointtsp.gif) | ![](https://github.com/andrewli77/combinatorial-rl-tasks/blob/master/gifs/timedtsp.gif) | ![](https://github.com/andrewli77/combinatorial-rl-tasks/blob/master/gifs/colourmatch.gif)
 

We propose a suite of lightweight robotics tasks based on combinatorial optimization problems that challenge state-of-the-art RL and hierarchical RL methods. These tasks have the following properties:

- **Combinatorally large solution spaces:** Each task can be solved in many ways, but identifying the optimal solution is hard. We expect that an agent's performance is tied to its *long-term reasoning ability*.
- **Dense rewards**: Tasks can be decomposed into dense reward formulations, enabling standard RL methods to learn without specialized exploration.



# Tasks

The code for our environments can be found in [`/main/envs`](/main/envs/). In all tasks, the observation space includes the agent's position and velocity, the position of zones (as an *unordered list*), and the time remaining in the episode.

### PointTSP
Based on the NP-hard Travelling Salesman Problem, the goal of this task is to visit all zones as quickly as possible. A dense reward of 1 is obtained by visiting a new city, and a sparse reward based on time remaining is obtained upon completing the task. This reward incentivizes completing the task above all else, and secondarily, to complete the task as fast as possible. 

### TimedTSP
This is a more challenging version of PointTSP, where each zone is randomly initialized with a timeout (observable to the agent). If any zone reaches its timeout before it is visited, the episode ends in failure. Therefore, the agent must prioritize zones close to timing out. If a timeout failure is imminent, the agent should visit as many zones as possible before the episode ends. 

### ColourMatch
The environment contains six zones with a randomly initialized colour (red, blue, or green; observable to the agent). Visiting a zone changes its colour to the next in a fixed cycle and the goal is to make all colours the same as quickly as possible. A dense reward is provided upon visiting any zone based on a discrete Hamming distance (minimum number of swaps to solve the task), and a sparse reward based on time remaining is obtained upon completing the episode. Note that all *successful* trajectories receive the same total dense reward, therefore the dense reward does not bias the policy towards any particular solution -- however, the sparse reward incentivizes the solution achievable in the shortest time. 

# Getting Started
## Installation
Please install the following packages before running the code. We recommend using **Python 3.6**. 

- [MuJoCo](https://mujoco.org/)
- [OpenAI Safety Gym](https://github.com/openai/safety-gym).
    - The install location should be the same directory as this README file. 
- A list of versions for major dependencies is in [`requirements.txt`](requirements.txt)


## Training and Evaluating RL agents
The code is separated into folders for different implemented methods. Instructions for training and evaluating each method can be found in that folder.

- To run **PPO**, **PPO-VD (ours)**, **Fixed-length skills**, **DIAYN**, or a **TSP-solver**-augmented approach, see the [`/main`](/main) folder. 
- To run **Zone-goals (ours)**, go to the [`/zone-goals`](/zone-goals) folder.
- To run variable-length **Options**, go to the [`/options`](/options) folder. 
- To run Hierarchical RL with **xy-goals**, go to the [`/xy-goals`](/xy-goals) folder.


# Tips and Best Practices

## Neural Architecture 
The positions of zones is observed as an *unordered list*, and it is highly recommended that you exploit this order-invariance property for sample-efficient learning. Our methods all encode this bias (see `ZoneEnvModel` in [`/main/src/env_model.py`](/main/src/env_model.py) for an example). Note that concatenating the list of zone observations into a vector *is not* order-invariant.

## Evaluation
**Evaluation should always use the undiscounted ($\gamma = 1$) episodic return** to ensure the evaluation objective matches the stated goal of the task (e.g. visiting all zones as fast as possible). In the paper, evaluation is performed across the same 100 randomly generated maps (environment seeds 1000000 - 1000099; see [`/main/scripts/evaluate.py`](/main/scripts/evaluate.py)). The performance of various methods can be found below (with training and evaluation details in the paper).

    
| Method                   | PointTSP  | TimedTSP | ColourMatch   |
| :----------------------: | :-------: | :------: | :-----------: |
| PPO ($\gamma=0.99$)      | 23.48     | 14.15    | 16.45         | 
| PPO ($\gamma=1$)         | 20.35     | 8.37     | 15.55         |
| PPO-VD                   | 23.36     | 16.19    | 16.33         |
| Fixed-length Skills      | 23.64     | 14.17    | 16.42         |
| Skills + Diversity       | 18.15     | 11.50    | 0.64          | 
| Options                  | 23.82     | 15.31    | 15.46         |
| xy-goals                 | 6.44      | 2.07     | 0.64          |
| Zone-goals               | 24.24     | 21.95    | 18.95         |
| Solver                   | 25.30     | -        | -             |
    


# Citation


Additional details, results, and insights can be found in the following paper: [Challenges to Solving Combinatorially Hard Long-Horizon Deep RL Tasks](https://arxiv.org/pdf/2206.01812.pdf) by Andrew C. Li, Pashootan Vaezipoor, Rodrigo Toro Icarte, Sheila A. McIlraith. 

**Bibtex:** 
``` 
@article{li2022challenges,
  title={Challenges to Solving Combinatorially Hard Long-Horizon Deep RL Tasks},
  author={Li, Andrew C. and Vaezipoor, Pashootan and Toro Icarte, Rodrigo and McIlraith, Sheila A.},
  journal={arXiv preprint arXiv:2206.01812},
  year={2022}
}
```
