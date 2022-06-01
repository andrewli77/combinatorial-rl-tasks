from multiprocessing import Process, Pipe
import gym

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "step_no_reset":
            obs, reward, done, info = env.step(data)
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "set_goal":
            env.set_goal(data)
        elif cmd == "get_goal":
            conn.send(env.get_goal())
        elif cmd == "needs_goal":
            conn.send(env.goal_zone == None)
        elif cmd == "available_goals":
            conn.send(env.get_available_goals())
        elif cmd == "kill":
            return
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def __del__(self):
        for local in self.locals:
            local.send(("kill", None))

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def step_no_reset(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step_no_reset", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def set_goal(self, env_idx, goal):
        if env_idx == 0:
            self.envs[0].set_goal(goal)
        else:
            self.locals[env_idx-1].send(("set_goal", goal))

    def get_goal(self, env_idx):
        if env_idx == 0:
            return self.envs[0].get_goal()
        else:
            self.locals[env_idx-1].send(("get_goal", None))
            return self.locals[env_idx-1].recv()

    def needs_goal(self):
        for local in self.locals:
            local.send(("needs_goal", None))
        return [self.envs[0].goal_zone == None] + [local.recv() for local in self.locals]

    def available_goals(self, env_idx):
        if env_idx == 0:
            return self.envs[0].get_available_goals()
        else:
            self.locals[env_idx-1].send(("available_goals", None))
            return self.locals[env_idx-1].recv()

    def render(self):
        raise NotImplementedError