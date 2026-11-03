"""Single command to reproduce all CW1 numerical outputs on an ordinary laptop."""

import argparse
from pathlib import Path
import sys

# Permit direct execution from any working directory without an editable install.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from sc_cw1.experiments import run_all  # noqa: E402


def main():
    """Parse the results destination and run every implemented coursework part."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT/"results")
    args = parser.parse_args()
    run_all(args.output)


if __name__ == "__main__":
    main()
