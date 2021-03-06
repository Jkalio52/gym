import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)
        self.finalize()

    def _step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def _reset(self):
        self.model.data.qpos = self.init_qpos + np.random.uniform(size=(self.model.nq,1), low=-0.01, high=0.01)
        self.model.data.qvel = self.init_qvel + np.random.uniform(size=(self.model.nv,1), low=-0.01, high=0.01)
        self.reset_viewer_if_necessary()        
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid=0
        v.cam.distance = v.model.stat.extent
