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
  # Periodic parameter resets (SPR/SR-SPR-style high-UTD intervention).
  should_reset = elements.when.Every(args.reset_every, initial=False)
  reset_count = 0
  # Meta-gradient train-ratio controller cadence (Tier 4a; schedule: meta_grad).
  should_meta = elements.when.Every(
      max(1, int(getattr(args, 'train_ratio_meta_every', 2000))), initial=False)

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

  # Tier-4a meta-gradient controller: a SEPARATE held-out stream/carry so the
  # controller can read its objective on a fast step-based cadence (reusing
  # agent.report(), keeping only the scalars) decoupled from the slow report above.
  meta_on = getattr(args, 'train_ratio_schedule', 'constant') == 'meta_grad'
  meta_warmup = int(getattr(args, 'train_ratio_meta_warmup', 3000))
  meta_batches = max(1, int(getattr(args, 'train_ratio_meta_batches', 1)))
  stream_meta = iter(agent.stream(make_stream(replay, 'report'))) if meta_on else None
  carry_meta = agent.init_report(args.batch_size) if meta_on else None

  # Dynamic train ratio state
  dyn_ratio_state = {'ema_loss': None, 'max_loss': 0.0}
  # Meta-gradient train-ratio state (schedule: meta_grad). The controller probes
  # the (log) ratio with an alternating +/- dither and estimates the hypergradient
  # by a DRIFT-ROBUST REGRESSION over a sliding window of (probe sign, held-out
  # objective): obj ~ b0 + b_time*time + b_sign*sign. The smooth, large training
  # drift (return naturally rises over a run) is absorbed by the time term; the
  # probe effect is the sign coefficient -- a high-frequency alternation orthogonal
  # to the trend -- so b_sign is an UNBIASED gradient regardless of drift magnitude.
  # (A plain two-point level difference does NOT cancel the drift; it injects an
  # alternating-sign drift term -> the random-signed gradient seen before. 2026-06-17
  # audit + Gemini review.) The step is normalized by RESIDUAL NOISE (not the
  # objective level, which would shrink steps for large-return games).
  meta_ratio_state = {
      'log_center': None,   # slowly-updated best-estimate log-ratio
      'cur_sign': 1,        # probe sign applied for the COMING window (+1/-1)
      'primed': False,      # has a probe been applied? (the first call measures the
                            # un-probed init window -- it carries no clean sign)
      'hist': [],           # sliding window of (sign, objective) over probe windows
      'grad': 0.0, 'updates': 0}

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    
    if getattr(args, 'train_ratio_schedule', 'constant') == 'linear':
      warmup = getattr(args, 'train_ratio_warmup_steps', 0)
      if warmup > 0:
        progress = min(1.0, int(step) / warmup)
      else:
        progress = int(step) / args.steps
      new_ratio = args.train_ratio_min + progress * (args.train_ratio_max - args.train_ratio_min)
      should_train._ratio = new_ratio / batch_steps

    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
        
      if getattr(args, 'train_ratio_schedule', 'constant') == 'dyn_loss_relative':
        if 'loss/dyn' in mets:
          current_loss = float(np.mean(mets['loss/dyn']))
          if dyn_ratio_state['ema_loss'] is None:
            dyn_ratio_state['ema_loss'] = current_loss
          else:
            alpha = getattr(args, 'train_ratio_ema_alpha', 0.99)
            dyn_ratio_state['ema_loss'] = alpha * dyn_ratio_state['ema_loss'] + (1 - alpha) * current_loss
          
          if dyn_ratio_state['ema_loss'] > dyn_ratio_state['max_loss']:
            dyn_ratio_state['max_loss'] = dyn_ratio_state['ema_loss']
          
          if dyn_ratio_state['max_loss'] > 0:
            rel = dyn_ratio_state['ema_loss'] / dyn_ratio_state['max_loss']
            new_ratio = args.train_ratio_min + (args.train_ratio_max - args.train_ratio_min) * (1.0 - rel)
            new_ratio = max(args.train_ratio_min, min(args.train_ratio_max, new_ratio))
            should_train._ratio = new_ratio / batch_steps

      # Log the current actual ratio being used
      train_agg.add({'train_ratio': should_train._ratio * batch_steps}, prefix='train')
      train_agg.add(mets, prefix='train')
      
  driver.on_step(trainfn)

  def metafn(o):
    # Drift-robust hypergradient on the (log) train ratio (Tier 4a). `o` is the RAW
    # held-out objective from the window that just ended, already sign-adjusted so
    # we always MINIMIZE it (o = -val for return objectives). We keep a sliding
    # window of (probe sign, o) and regress o ~ b0 + b_time*time + b_sign*sign:
    # the smooth training drift is absorbed by b_time, the probe effect is b_sign
    # (alternation orthogonal to the trend), so the gradient dObj/d(log ratio) =
    # b_sign/dither is UNBIASED by drift. We step log_center downhill, normalizing
    # by RESIDUAL NOISE (scale-free without shrinking steps for large-return tasks)
    # and clipping the log-space step so one noisy estimate can't slam to a bound.
    s = meta_ratio_state
    if not np.isfinite(o):
      return
    lo, hi = float(args.train_ratio_min), float(args.train_ratio_max)
    loglo, loghi = np.log(lo), np.log(hi)
    meta_lr = getattr(args, 'train_ratio_meta_lr', 0.3)
    cost = getattr(args, 'train_ratio_meta_cost', 0.0)
    dither = getattr(args, 'train_ratio_meta_dither', 0.3)
    step_clip = getattr(args, 'train_ratio_meta_step_clip', 0.1)
    window = int(getattr(args, 'train_ratio_meta_window', 12))
    if s['log_center'] is None:
      cur = should_train._ratio * batch_steps
      s['log_center'] = float(np.clip(np.log(max(cur, 1e-3)), loglo, loghi))
    # Record the just-ended window's (probe sign, objective). The very first call
    # measured the un-probed init window (cur_sign has no probe behind it yet), so
    # skip recording until a probe has actually been applied.
    if s['primed']:
      s['hist'].append((float(s['cur_sign']), float(o)))
      s['hist'] = s['hist'][-window:]
      signs = np.array([h[0] for h in s['hist']])
      if len(s['hist']) >= 6 and signs.std() > 0:
        y = np.array([h[1] for h in s['hist']])
        t = np.arange(len(y), dtype=np.float64)
        t -= t.mean()
        A = np.stack([np.ones_like(t), t, signs], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        beta_s = float(coef[2])
        noise = float((y - A @ coef).std()) + 1e-8
        s['grad'] = beta_s / dither  # dObj/d(log ratio), for logging
        # Descend o; step ~ SNR (probe amplitude in noise units), clipped.
        delta = float(np.clip(meta_lr * beta_s / noise, -step_clip, step_clip))
        s['log_center'] = float(np.clip(
            s['log_center'] - delta - cost, loglo, loghi))
    # Flip to the other side and apply the new probe for the coming window.
    s['cur_sign'] *= -1
    s['primed'] = True
    log_ratio = float(np.clip(s['log_center'] + s['cur_sign'] * dither, loglo, loghi))
    new_ratio = float(np.exp(log_ratio))
    should_train._ratio = new_ratio / batch_steps
    s['updates'] += 1
    train_agg.add({'train_ratio': new_ratio,
                   'meta_ratio_grad': s['grad'],
                   'meta_ratio_obj': o,
                   'meta_ratio_center': float(np.exp(s['log_center']))}, prefix='train')
    print(f'[meta_grad] #{s["updates"]} step {int(step)} ratio={new_ratio:.1f} '
          f'center={float(np.exp(s["log_center"])):.1f} obj={o:.3f} '
          f'g={s["grad"]:+.4f}', flush=True)

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

    if args.reset_every and should_reset(step):
      agent.reset_params(args.reset_regex)
      reset_count += 1
      print(f'Periodic reset #{reset_count} ({args.reset_regex!r}) '
            f'at step {int(step)}')
      train_agg.add({'reset_count': reset_count}, prefix='train')

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      report_result = agg.result()
      logger.add(report_result, prefix='report')

    # Tier-4a meta-gradient controller (schedule: meta_grad), on its OWN step-based
    # cadence DECOUPLED from the slow report above. It runs a held-out eval to read
    # the meta-objective and takes one probe/meta-step -- giving the controller
    # hundreds of updates over a run instead of the ~5 it got when gated behind the
    # 20-min report (audit #2). We reuse the (already jitted) agent.report() but
    # keep ONLY the held-out scalars (loss/*, ret, val) -- the openloop video is
    # neither aggregated nor logged here, so there is no encode/upload cost.
    if meta_on and len(replay) and int(step) >= meta_warmup and should_meta(step):
      agg = elements.Agg()
      for _ in range(meta_batches):
        carry_meta, mets = agent.report(carry_meta, next(stream_meta))
        agg.add({k: v for k, v in mets.items() if isinstance(k, str)
                 and (k.startswith('loss/') or k in ('ret', 'val'))})
      mres = agg.result()
      okey = getattr(args, 'train_ratio_meta_objective', 'val')
      oval = mres.get(okey)
      # Maximize held-out return (val/ret) -> minimize its negation; or minimize a
      # held-out loss objective directly.
      sign = -1.0 if okey in ('val', 'ret') else 1.0
      if oval is None:  # coherent fallback: world-model losses only (all
        # 'lower=better'; AC/aux losses are normalized/can be negative -> excluded).
        skip = ('policy', 'value', 'repval', 'bc_loss', 'disag')
        wm = [v for k, v in mres.items() if isinstance(k, str)
              and k.startswith('loss/') and k.rsplit('/', 1)[-1] not in skip]
        oval, sign = (float(np.sum(wm)) if wm else None), 1.0
      if oval is not None and np.isfinite(float(oval)):
        metafn(sign * float(oval))

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
