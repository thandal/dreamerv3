"""Tests for multi-task wrappers and utilities."""

import elements
import embodied
import numpy as np
import pytest


class DummyDiscEnv(embodied.Env):
  """Minimal discrete-action env for testing wrappers."""

  def __init__(self, num_actions=5, image_size=(8, 8)):
    self._num_actions = num_actions
    self._image_size = image_size
    self._done = True

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, (*self._image_size, 3)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._num_actions),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      return self._obs(0.0, is_first=True)
    self._done = True
    return self._obs(1.0, is_last=True, is_terminal=True)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    return dict(
        image=np.zeros((*self._image_size, 3), np.uint8),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )


class TestUnifyActions:

  def test_pads_action_space(self):
    env = DummyDiscEnv(num_actions=5)
    wrapped = embodied.multitask.UnifyActions(env, key='action', max_actions=18)
    assert wrapped.act_space['action'].high == 18
    assert wrapped.act_space['action'].low == 0

  def test_valid_action_passes_through(self):
    env = DummyDiscEnv(num_actions=5)
    wrapped = embodied.multitask.UnifyActions(env, key='action', max_actions=18)
    obs = wrapped.step({'action': np.int32(3), 'reset': True})
    assert obs['is_first']

  def test_out_of_range_action_maps_to_noop(self):
    env = DummyDiscEnv(num_actions=5)
    wrapped = embodied.multitask.UnifyActions(env, key='action', max_actions=18)
    # First reset
    wrapped.step({'action': np.int32(0), 'reset': True})
    # Send an action beyond valid range — should map to 0 (NOOP)
    obs = wrapped.step({'action': np.int32(15), 'reset': False})
    # Should not error — action 0 is always valid
    assert obs is not None

  def test_boundary_action(self):
    env = DummyDiscEnv(num_actions=5)
    wrapped = embodied.multitask.UnifyActions(env, key='action', max_actions=18)
    wrapped.step({'action': np.int32(0), 'reset': True})
    # Action 4 is last valid action for 5-action env
    obs = wrapped.step({'action': np.int32(4), 'reset': False})
    assert obs is not None
    # Action 5 should map to NOOP
    wrapped.step({'action': np.int32(0), 'reset': True})
    obs = wrapped.step({'action': np.int32(5), 'reset': False})
    assert obs is not None

  def test_preserves_obs_space(self):
    env = DummyDiscEnv(num_actions=5)
    wrapped = embodied.multitask.UnifyActions(env, key='action', max_actions=18)
    assert wrapped.obs_space == env.obs_space

  def test_error_if_max_too_small(self):
    env = DummyDiscEnv(num_actions=10)
    with pytest.raises(AssertionError):
      embodied.multitask.UnifyActions(env, key='action', max_actions=5)


class TestAddTaskID:

  def test_adds_task_id_to_obs_space(self):
    env = DummyDiscEnv()
    wrapped = embodied.multitask.AddTaskID(env, task_id=2, num_tasks=5)
    assert 'task_id' in wrapped.obs_space
    assert wrapped.obs_space['task_id'].high == 5
    assert wrapped.obs_space['task_id'].low == 0

  def test_task_id_in_observations(self):
    env = DummyDiscEnv()
    wrapped = embodied.multitask.AddTaskID(env, task_id=3, num_tasks=5)
    obs = wrapped.step({'action': np.int32(0), 'reset': True})
    assert obs['task_id'] == 3

  def test_preserves_act_space(self):
    env = DummyDiscEnv()
    wrapped = embodied.multitask.AddTaskID(env, task_id=0, num_tasks=3)
    assert wrapped.act_space == env.act_space


class TestAssignTasks:

  def test_round_robin(self):
    tasks = ['task_a', 'task_b', 'task_c']
    assignments = embodied.multitask.assign_tasks(tasks, 6, 'round_robin')
    assert len(assignments) == 6
    assert assignments[0] == ('task_a', 0)
    assert assignments[1] == ('task_b', 1)
    assert assignments[2] == ('task_c', 2)
    assert assignments[3] == ('task_a', 0)
    assert assignments[4] == ('task_b', 1)
    assert assignments[5] == ('task_c', 2)

  def test_random_valid_assignments(self):
    tasks = ['task_a', 'task_b']
    assignments = embodied.multitask.assign_tasks(
        tasks, 10, 'random', seed=42)
    assert len(assignments) == 10
    for name, idx in assignments:
      assert name in tasks
      assert 0 <= idx < len(tasks)

  def test_random_deterministic(self):
    tasks = ['task_a', 'task_b', 'task_c']
    a1 = embodied.multitask.assign_tasks(tasks, 8, 'random', seed=123)
    a2 = embodied.multitask.assign_tasks(tasks, 8, 'random', seed=123)
    assert a1 == a2

  def test_invalid_strategy(self):
    with pytest.raises(ValueError):
      embodied.multitask.assign_tasks(['a'], 1, 'invalid')

  def test_single_task(self):
    assignments = embodied.multitask.assign_tasks(
        ['task_a'], 4, 'round_robin')
    assert all(name == 'task_a' for name, _ in assignments)
    assert all(idx == 0 for _, idx in assignments)


class TestCombinedWrappers:

  def test_unify_then_task_id(self):
    """Test that both wrappers compose correctly."""
    env = DummyDiscEnv(num_actions=4)
    env = embodied.multitask.UnifyActions(env, key='action', max_actions=18)
    env = embodied.multitask.AddTaskID(env, task_id=1, num_tasks=3)

    assert env.act_space['action'].high == 18
    assert 'task_id' in env.obs_space

    obs = env.step({'action': np.int32(0), 'reset': True})
    assert obs['task_id'] == 1
    assert obs['is_first']

  def test_different_envs_same_space(self):
    """Multiple envs with different action counts get unified."""
    envs = []
    for i, n_actions in enumerate([3, 6, 10]):
      env = DummyDiscEnv(num_actions=n_actions)
      env = embodied.multitask.UnifyActions(
          env, key='action', max_actions=18)
      env = embodied.multitask.AddTaskID(env, task_id=i, num_tasks=3)
      envs.append(env)

    # All should have same act/obs spaces
    for env in envs[1:]:
      assert env.act_space == envs[0].act_space
      assert env.obs_space.keys() == envs[0].obs_space.keys()
