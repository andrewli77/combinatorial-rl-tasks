Make sure to run `setup.sh` from this directory to load all the proper paths! 

=======================================================
Training commands:
=======================================================

We provide the script to train each agent on the PointTSP environment. For other environments, replace PointTSP with PointTTSP (for TimedTSP) or ColourMatch. You may wish to change the hyperparameters to those reported in the supplementary material. 

- PPO: `python3 scripts/train_ppo.py --env PointTSP-v0 --frames 100000000 --discount 0.99 --frames-per-proc 4000`

- PPO-VD: `python3 scripts/train_ppo.py --env PointTSP-v0 --frames 100000000 --discount 1. --frames-per-proc 4000 --distributional-value --value-loss-coef 0.005 --epochs 6`

- Fixed-length skills: `python3 scripts/train_skill_planner.py --env PointTSP-v0 --frames 100000000 --diversity-coef 0. --train-lo --train-hi --n-skills 5 --hi-entropy-coef 0.01 --clip-eps 0.1 --frames-per-proc 4000`

- DIAYN: `python3 scripts/train_skill_planner.py --env PointTSP-v0 --frames 100000000 --diversity-coef 0.01 --train-lo --train-hi --n-skills 5 --hi-entropy-coef 0.01 --clip-eps 0.1 --frames-per-proc 4000`

- TSP-solver: `python3 scripts/train_ppo.py --env PointTSP-v2 --frames 100000000 --discount 0.99 --frames-per-proc 4000`

=======================================================
Evaluation commands:
=======================================================

Replace `model_location` with the location of the trained model you want to evaluate. 

- PPO: `python3 scripts/evaluate.py --env PointTSP-v0 --model model_location/`

- PPO-VD: `python3 scripts/evaluate.py --env PointTSP-v0 --distributional-value --model model_location/`

- Fixed-length-skills, DIAYN: `python3 scripts/evaluate_hier.py --env PointTSP-v0 --model model_location/`

- TSP-solver: `python3 scripts/evaluate.py --env PointTSP-v21 --model model_location/`