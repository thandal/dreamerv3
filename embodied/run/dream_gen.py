from functools import partial as bind

import elements
import embodied
import numpy as np

from embodied.core import chunk as chunklib


def dream_gen(make_agent, make_replay, make_stream, make_logger, args):
  """Step-3 generative replay: imagine actor-driven rollouts from a small seed bank
  through a (frozen) checkpointed world model, decode them to synthetic observations,
  and write them as replay chunks. The sleep/consolidation phase then trains on these
  WM-GENERATED chunks (via --run.replay_dirs) instead of the stored real buffer.

  Uses the agent's report() entry-point in dream mode (config.dream_gen=True), which
  returns the dreamed trajectory (image/action/reward/cont + RSSM latents) as arrays.
  """
  assert args.from_checkpoint, 'dream_gen needs --run.from_checkpoint'
  agent = make_agent()
  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  dreamdir = logdir / 'dreams'
  dreamdir.mkdir()
  print('Dream output dir:', dreamdir)

  # Load the frozen generator agent (the peak-snapshot world model).
  elements.checkpoint.load(args.from_checkpoint, dict(
      agent=bind(agent.load, regex=args.from_checkpoint_regex)))

  # Seed bank: load the small set of real chunks we imagine from.
  replay = make_replay()
  for d in (getattr(args, 'replay_dirs', '') or '').replace(',', ' ').split():
    print('Loading dream seeds from', d)
    replay.load(directory=d)
  assert len(replay), 'no seed data loaded for dream_gen'
  # report() is AOT-compiled for the REPORT-stream shape, so feed it that (not 'train').
  stream = iter(agent.stream(make_stream(replay, 'report')))

  def act_index(a):
    a = np.asarray(a)
    return a.argmax(-1).astype(np.int32) if a.ndim == 3 else a.astype(np.int32)

  carry = agent.init_report(args.batch_size)
  target = int(args.steps)   # number of dreamed STEPS to generate
  written, gidx = 0, 0
  while written < target:
    carry, out = agent.report(carry, next(stream))
    img = np.asarray(out['dream/image'])       # (BK, H, ...)
    rew = np.asarray(out['dream/reward'])       # (BK, H)
    cont = np.asarray(out['dream/cont'])        # (BK, H)
    deter = np.asarray(out['dream/deter'], np.float32)
    stoch = np.asarray(out['dream/stoch'], np.float32)
    action = act_index(out['dream/act/action'])  # (BK, H)
    BK, H = rew.shape[:2]
    for i in range(BK):
      ch = chunklib.Chunk(H)
      for h in range(H):
        ch.append({
            'image': np.asarray(img[i, h]),
            'reward': np.float32(rew[i, h]),
            'is_first': np.bool_(h == 0),
            'is_last': np.bool_(h == H - 1),
            'is_terminal': np.bool_(cont[i, h] < 0.5),
            'action': np.int32(action[i, h]),
            'dyn/deter': np.asarray(deter[i, h], np.float32),
            'dyn/stoch': np.asarray(stoch[i, h], np.float32),
            'stepid': np.frombuffer(gidx.to_bytes(20, 'big'), np.uint8).copy(),
        })
        gidx += 1
      ch.save(str(dreamdir))
      written += H
    # Dream-quality diagnostics: degenerate reward stats (all-zero, exploding) or
    # collapsed cont indicate a broken generator — surface them in the stage log.
    print(f'dreamed {written}/{target} steps | rew mean={rew.mean():.4f} sd={rew.std():.4f} '
          f'max={rew.max():.2f} | cont mean={cont.mean():.3f} term_frac={(cont < 0.5).mean():.3f}',
          flush=True)
  print(f'DREAM_GEN DONE: {written} steps -> {dreamdir}', flush=True)
