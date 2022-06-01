from envs.wrappers import *

def make_train_env(env_name, hier=False, num_training_tasks = 100, rng_seed = 0):
    env = gym.make(env_name)

    if env_name in ['PointPush-v0', 'CarPush-v0', 'DoggoPush-v0', 'CarPush-v1',
        'PointGoal-v0', 'CarGoal-v0']:
        if hier:
            return WaitWrapper(HierWrapper(FixedSeedsWrapper(env, min_seed=1, max_seed=num_training_tasks, rng_seed=rng_seed)))
        else:
            return FixedSeedsWrapper(env, min_seed=1, max_seed=num_training_tasks, rng_seed=rng_seed)
    elif env_name in ['PointTSP-v0', 'PointTSP-v1', 'PointTSP-v2', 'PointTSP-v3', 'PointTTSP-v0', 'PointTTSP-v1', 'CarTSP-v0', 'DoggoTSP-v0', 'ColourMatch-v0']:
        if hier:
            return WaitWrapper(ZoneWrapper(FixedSeedsWrapper(env, min_seed=1, max_seed=num_training_tasks, rng_seed=rng_seed)))
        else:
            return ZoneWrapper(FixedSeedsWrapper(env, min_seed=1, max_seed=num_training_tasks, rng_seed=rng_seed))
    else:
        raise RuntimeError("Unknown environment")

def make_test_env(env_name, hier=False, seed=1000):
    env = gym.make(env_name)
    env.seed(seed)

    if env_name in ['PointPush-v0', 'CarPush-v0', 'DoggoPush-v0', 'CarPush-v1',
        'PointGoal-v0', 'CarGoal-v0']:
        if hier:
            return HierWrapper(env)
        else:
            return env

    elif env_name in ['PointTSP-v0', 'PointTSP-v1', 'PointTSP-v2', 'PointTSP-v3', 'PointTSP-v4', 'PointTSP-v5', 'PointTTSP-v0', 'PointTTSP-v1', 'CarTSP-v0', 'DoggoTSP-v0', 'ColourMatch-v0']:
        return ZoneWrapper(env)
    else:
        raise RuntimeError("Unknown environment")


def make_fixed_env(env_name, hier=False, seed=1000, env_seed=0):
    env = gym.make(env_name)
    env.seed(seed)

    if env_name in ['PointPush-v0', 'CarPush-v0', 'DoggoPush-v0', 'CarPush-v1',
        'PointGoal-v0', 'CarGoal-v0']:
        if hier:
            return HierWrapper(FixedSeedsWrapper(env, min_seed=env_seed, max_seed=env_seed, rng_seed=seed))
        else:
            return FixedSeedsWrapper(env, min_seed=env_seed, max_seed=env_seed, rng_seed=seed)

    elif env_name in ['PointTSP-v0', 'PointTSP-v1', 'PointTSP-v2', 'PointTSP-v3', 'PointTSP-v4', 'PointTSP-v5', 'PointTTSP-v0', 'PointTTSP-v1', 'CarTSP-v0', 'DoggoTSP-v0', 'ColourMatch-v0']:
        return ZoneWrapper(FixedSeedsWrapper(env, min_seed=env_seed, max_seed=env_seed, rng_seed=seed))
    else:
        raise RuntimeError("Unknown environment")