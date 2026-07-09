import os
import sys

import elements
import embodied
import numpy as np


class Excavator(embodied.Env):
  """Mini-excavator earthmoving tasks (dig/trench/grade/backfill/...).

  Thin adapter over the framework-free env in the remodext repo
  (excavator/excavator_env.py): pybullet arm + deformable soil grid +
  skid-steer base, domain-randomized per episode. Reward is design-surface
  progress (cut+fill deficit reduction). Tasks: dig_trench, load_pile,
  curved_trench, dig_pad, grade_lane, grade_crowned, slope_cut,
  backfill_trench, consolidate_piles.

  Config: env.excavator.repo points at the remodext excavator directory
  (default ~/code/remodext/excavator, override with $EXCAVATOR_REPO).
  """

  def __init__(self, task, repo='', length=1800, noise=True, size=64, seed=0,
               shaping=5.0, reward_mode='dmc', objective='full'):
    repo = (repo or os.environ.get('EXCAVATOR_REPO')
            or os.path.expanduser('~/code/remodext/excavator'))
    if repo not in sys.path:
      sys.path.insert(0, repo)
    from excavator_env import ACTION_DIM, VECTOR_DIM, ExcavatorEnv
    self._vector_dim = VECTOR_DIM
    self._action_dim = ACTION_DIM
    self._size = size
    # Curriculum rungs as task-name prefixes: excavator_scoop.dig_trench ->
    # objective='scoop' on the dig_trench terrain.
    for rung in ('scoop', 'cycle'):
      if task.startswith(rung + '.'):
        objective, task = rung, task[len(rung) + 1:]
    self._env = ExcavatorEnv(task=task, length=length, noise=noise,
                             size=size, seed=seed, shaping=shaping,
                             reward_mode=reward_mode, objective=objective)
    self._done = True

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, (self._size, self._size, 3)),
        'vector': elements.Space(np.float32, (self._vector_dim,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.float32, (self._action_dim,), -1.0, 1.0),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      obs = self._env.reset()
      self._done = False
      return self._obs(obs, 0.0, is_first=True)
    obs, reward, done, terminal = self._env.step(action['action'])
    self._done = done
    return self._obs(obs, reward, is_last=done, is_terminal=terminal)

  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    return dict(
        image=obs['image'],
        vector=obs['vector'],
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )

  def close(self):
    self._env.close()
