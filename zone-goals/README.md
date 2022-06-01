Make sure to run `setup.sh` from this directory to load all the proper paths! 

=======================================================
Training commands:
=======================================================

We provide the script to train each agent on the PointTSP environment. For other environments, replace PointTSP with PointTTSP (for TimedTSP) or ColourMatch. You may wish to change the hyperparameters to those reported in the supplementary material. 

- Zone-goals: `python3 scripts/train_skill_planner.py --env PointTSP-v3 --frames 100000000 --discount 0.99 --frames-per-proc 4000 --hi-entropy-coef 0.01 --clip-eps 0.1`


=======================================================
Evaluation commands:
=======================================================

Replace `model_location` with the location of the trained model you want to evaluate. 

- Zone-goals: `python3 scripts/evaluate.py --env PointTSP-v3 --model model_location/`
