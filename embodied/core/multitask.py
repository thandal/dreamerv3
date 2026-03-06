"""Multi-task wrappers and utilities for training across multiple environments."""

import functools

import elements
import numpy as np

from .wrappers import Wrapper


class UnifyActions(Wrapper):
  """Pad a discrete action space to a fixed size.

  Actions beyond the environment's real action count are mapped to action 0
  (typically NOOP). This ensures all tasks present the same action space to
  the agent.
  """

  def __init__(self, env, key='action', max_actions=18):
    super().__init__(env)
    self._key = key
    self._real_n = env.act_space[key].high
    self._max_actions = max_actions
    assert env.act_space[key].discrete, (
        f'UnifyActions only works with discrete action spaces, '
        f'got {env.act_space[key]}')
    assert self._real_n <= max_actions, (
        f'Environment has {self._real_n} actions but max_actions={max_actions}')

  @functools.cached_property
  def act_space(self):
    space = elements.Space(np.int32, (), 0, self._max_actions)
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    action = action.copy()
    act = action[self._key]
    # Map out-of-range actions to 0 (NOOP)
    if act >= self._real_n:
      act = np.int32(0)
    action[self._key] = act
    return self.env.step(action)


class UnifyContinuousActions(Wrapper):
  """Pad a continuous action space to a fixed dimensionality.

  Actions beyond the environment's real dimensionality are ignored.
  This ensures all tasks present the same action shape to the agent.
  The padded space uses [-1, 1] bounds (assumes NormalizeAction is applied).
  """

  def __init__(self, env, key='action', max_dim=6):
    super().__init__(env)
    self._key = key
    space = env.act_space[key]
    assert not space.discrete, (
        f'UnifyContinuousActions only works with continuous action spaces, '
        f'got {space}')
    self._real_dim = space.shape[0]
    self._max_dim = max_dim
    assert self._real_dim <= max_dim, (
        f'Environment has {self._real_dim} action dims but max_dim={max_dim}')

  @functools.cached_property
  def act_space(self):
    space = elements.Space(
        np.float32, (self._max_dim,),
        -np.ones(self._max_dim), np.ones(self._max_dim))
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    action = action.copy()
    # Slice out only the real action dimensions, discard padding
    action[self._key] = action[self._key][:self._real_dim]
    return self.env.step(action)


class AddTaskID(Wrapper):
  """Add a task_id integer observation to every step.

  The task_id is a constant integer identifying which task this environment
  represents. Combined with num_tasks, it allows the agent to condition on
  task identity.
  """

  def __init__(self, env, task_id, num_tasks):
    super().__init__(env)
    self._task_id = np.int32(task_id)
    self._num_tasks = num_tasks

  @functools.cached_property
  def obs_space(self):
    return {
        **self.env.obs_space,
        'task_id': elements.Space(np.int32, (), 0, self._num_tasks),
    }

  def step(self, action):
    obs = self.env.step(action)
    obs['task_id'] = self._task_id
    return obs


def assign_tasks(tasks, num_workers, strategy='round_robin', seed=None):
  """Assign tasks to worker indices based on a scheduling strategy.

  Args:
    tasks: List of task name strings.
    num_workers: Number of parallel env workers.
    strategy: 'round_robin' or 'random'.
    seed: Random seed (used only for 'random' strategy).

  Returns:
    List of (task_name, task_index) tuples, one per worker.
  """
  if strategy == 'round_robin':
    assignments = [
        (tasks[i % len(tasks)], i % len(tasks))
        for i in range(num_workers)]
  elif strategy == 'random':
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(tasks), size=num_workers)
    assignments = [(tasks[i], int(i)) for i in indices]
  else:
    raise ValueError(f'Unknown multi-task strategy: {strategy}')
  return assignments
