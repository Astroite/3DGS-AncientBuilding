"""Compare backends on a passed Run segment; never launch full-scene reconstruction."""
from pathlib import Path
import argparse

from gsdb.comparison import compare_backends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path, help='Directory containing the Run manifest.yaml')
    parser.add_argument('--segment', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--duration-seconds', type=float)
    parser.add_argument('--dry-run', action='store_true', help='Prepare inputs and commands without training')
    parser.add_argument('--resume', action='store_true', help='Resume native checkpoints; verify and reuse completed results')
    parser.add_argument('--backends', nargs='+', choices=['gsplat', 'postshot'], default=['postshot', 'gsplat'])
    parser.add_argument('--photo-comp', choices=['both', 'on', 'off'], default='both')
    args = parser.parse_args()
    try:
        result = compare_backends(args.run_dir, args.segment, args.output, backends=args.backends,
            photo_comp={'both': (False, True), 'on': (True,), 'off': (False,)}[args.photo_comp],
            steps=args.steps, duration_seconds=args.duration_seconds, resume=args.resume, dry_run=args.dry_run)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Comparison failed: {error}\n')
    if any(e['status'] in {'failed', 'blocked'} for e in result['experiments']):
        parser.exit(1, 'Comparison contains failed or blocked experiments; inspect comparison.json\n')


if __name__ == '__main__':
    main()
