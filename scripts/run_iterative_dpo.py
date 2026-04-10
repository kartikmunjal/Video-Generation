"""
Run the iterative DiffusionDPO alignment loop.

Usage:
    python scripts/run_iterative_dpo.py --config configs/iterative_dpo_config.yaml
    python scripts/run_iterative_dpo.py --config configs/iterative_dpo_config.yaml \
        --max_iterations 3 --buffer_strategy rolling2
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.iterative_dpo import IterativeDPOLoop


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/iterative_dpo_config.yaml")
    p.add_argument("--max_iterations", type=int, default=None)
    p.add_argument("--target_win_rate", type=float, default=None)
    p.add_argument("--buffer_strategy", default=None, choices=["current", "rolling2", "full"])
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.max_iterations is not None:
        config["loop"]["max_iterations"] = args.max_iterations
    if args.target_win_rate is not None:
        config["loop"]["target_win_rate"] = args.target_win_rate
    if args.buffer_strategy is not None:
        config["buffer"]["strategy"] = args.buffer_strategy
    if args.output_dir is not None:
        config["output"]["base_dir"] = args.output_dir

    print("Iterative DiffusionDPO")
    print(f"  Max iterations: {config['loop']['max_iterations']}")
    print(f"  Target win rate: {config['loop']['target_win_rate']}")
    print(f"  Buffer strategy: {config['buffer']['strategy']}")
    print(f"  Output: {config['output']['base_dir']}")

    loop = IterativeDPOLoop(config)
    loop.setup()
    final_checkpoint, history = loop.run()

    print(f"\nFinal checkpoint: {final_checkpoint}")
    for h in history:
        print(f"  Iter {h['iteration']}: win_rate={h['win_rate']:.3f}, pairs={h['n_pairs']}")


if __name__ == "__main__":
    main()
