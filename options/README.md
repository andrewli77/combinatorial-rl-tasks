Make sure to run `setup.sh` from this directory to load all the proper paths! 

## Training commands

We provide the script to train each agent on the `PointTSP` environment. For other environments, replace `PointTSP` with `PointTTSP` (for `TimedTSP`) or `ColourMatch`. You may wish to change the hyperparameters to those reported in the supplementary material. 

- Options: 
    ```
    python3 scripts/train_skill_planner.py --env PointTSP-v0 --frames 100000000 --train-lo --train-hi --n-skills 5 --hi-entropy-coef 0.01 --clip-eps 0.1 --frames-per-proc 4000
    ```


## Evaluation commands

Replace `model_location` with the location of the trained model you want to evaluate. 

- Zone-goals: 
    ```
    python3 scripts/evaluate_hier.py --env PointTSP-v0 --model model_location/
    ```


