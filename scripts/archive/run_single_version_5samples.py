#!/usr/bin/env python3
from pathlib import Path
import argparse
from src.data.loader import DataLoader
from src.simulation.simulator import PilotSimulator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--version', required=True)
    p.add_argument('--model', default='gpt-5.1-mini')
    p.add_argument('--user', type=int, default=71)
    p.add_argument('--n', type=int, default=5)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--delay', type=float, default=0.5)
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(root / 'data')
    sim = PilotSimulator(
        loader=loader,
        output_dir=out,
        pilot_user_ids=[args.user],
        dry_run=False,
        model=args.model,
        agentic_model=args.model,
        delay=args.delay,
    )
    sim.setup()
    ud = sim._users_data[0]
    ud['ema_entries'] = ud['ema_entries'][:args.n]
    ud['sensing_days'] = ud['sensing_days'][:args.n]
    sim._users_data = [ud]

    res = sim.run_version(args.version)
    print('version', args.version, 'predictions', len(res['predictions']), 'llm_calls', res['total_llm_calls'])
    print('output_dir', out)


if __name__ == '__main__':
    main()
