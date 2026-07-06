import json
from functools import partial as bind

import elements
import numpy as np


def probe_competence(make_agent, make_stream, args, make_replay_at):
  """Step-3c cross-stage competence probe (env-free, no training).

  Load a checkpoint (a task's PEAK snapshot, e.g. G0 after the joint stage) and,
  for each replay dir in run.replay_dirs, measure the imagined lambda-return of
  the checkpoint's policy from that reservoir's states -- the same probe the
  competence-gated sleep runs, using 'ret_raw' (RAW reward scale, no return
  normalizer) so values are comparable across stages and checkpoints. Writes
  logdir/competence.json mapping each replay dir to its mean competence; sleep
  consumes these via run.sleep_best_from as LIFETIME-best recovery targets.
  """
  assert args.from_checkpoint, 'probe_competence needs --run.from_checkpoint'
  agent = make_agent()
  elements.checkpoint.load(args.from_checkpoint, dict(
      agent=bind(agent.load, regex=args.from_checkpoint_regex)))

  sources = (getattr(args, 'replay_dirs', '') or '').replace(',', ' ').split()
  assert sources, 'probe_competence needs --run.replay_dirs'
  batches = max(1, int(getattr(args, 'probe_batches', 8)))

  out = {}
  for i, d in enumerate(sources):
    r = make_replay_at(f'probe_src_{i}')
    r.load(directory=d)
    if not len(r):
      print(f'[probe_competence] WARNING: no data loaded from {d} -- skipping')
      out[d.rstrip('/')] = None
      continue
    stream = iter(agent.stream(make_stream(r, 'report')))
    carry = agent.init_report(args.batch_size)
    vals = []
    for _ in range(batches):
      carry, mets = agent.report(carry, next(stream))
      c = mets.get('ret_raw', mets.get('ret', mets.get('val')))
      if c is not None and np.isfinite(float(np.asarray(c))):
        vals.append(float(np.asarray(c)))
    out[d.rstrip('/')] = float(np.mean(vals)) if vals else None
    print(f'[probe_competence] {d}: ret_raw={out[d.rstrip("/")]} '
          f'(n={len(vals)}/{batches})', flush=True)

  path = elements.Path(args.logdir) / 'competence.json'
  path.write(json.dumps(out, indent=2))
  print(f'PROBE_COMPETENCE DONE: {len(sources)} reservoir(s) -> {path}', flush=True)
