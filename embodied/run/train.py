import collections
from functools import partial as bind

import elements
import embodied
import numpy as np


def train(
    make_agent, make_replay, make_env, make_stream, make_logger, args,
    task_assignments=None):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  # Per-task episode aggregators
  task_map = task_assignments or {}
  task_epstats = collections.defaultdict(elements.Agg) if task_map else {}
  task_latest = {}  # most recent episode stats per task, for computing mean

  # Determine which workers capture images (one per task, or worker 0)
  if task_map:
    image_workers = set()
    seen_tasks = set()
    for w in sorted(task_map):
      if task_map[w] not in seen_tasks:
        image_workers.add(w)
        seen_tasks.add(task_map[w])
  else:
    image_workers = {0}

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker in image_workers:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      score = result.pop('score')
      length = result.pop('length')
      rew = result.pop('rewards')
      reward_rate = None
      if len(rew) > 1:
        reward_rate = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
        result['reward_rate'] = reward_rate
      if not task_map:
        epstats.add(result)
        logger.add({'score': score, 'length': length}, prefix='episode')
      else:
        # Global epstats without images (per-task epstats have them)
        epstats.add({k: v for k, v in result.items()
                     if not k.startswith('policy_')})
        if worker in task_map:
          task_name = task_map[worker]
          logger.add({'score': score, 'length': length},
                     prefix=f'episode/{task_name}')
          task_epstats[task_name].add(result)
          task_latest[task_name] = {'score': score, 'length': length}
        if task_latest:
          mean_score = np.mean([v['score'] for v in task_latest.values()])
          mean_length = np.mean([v['length'] for v in task_latest.values()])
          logger.add({'score': mean_score, 'length': mean_length}, prefix='episode')

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      for task_name, agg in task_epstats.items():
        logger.add(agg.result(), prefix=f'epstats/{task_name}')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()
