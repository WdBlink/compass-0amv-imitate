"""Rescore immutable candidate revisions in a temporary clone, leaving the caller's tree alone."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[1]
EXTRA = Path.home() / 'Data/datasets/derived/compass-0amv/autoresearch-261121-2039'
INPUT_HASHES = {
    'blocks.npz': '6cadc8aa101d29047492a45b41b519d794a8804fa653c9a70bee6f5b8214d611',
    'lifecycle.npz': 'e8aaf0c268009d747900e7a1d646804bd68d3844a65d1998fc7f772c9020f4d8',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra', type=Path, default=EXTRA, help='local market-only supplemental input directory')
    args = parser.parse_args()
    for name, expected in INPUT_HASHES.items():
        if hashlib.sha256((args.extra / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'{name}: supplement differs from the frozen input')
    with (RUN / 'loop-results.tsv').open() as log:
        rows = list(csv.DictReader((line for line in log if not line.startswith('#')), delimiter='\t'))
    handoff = json.loads((RUN / 'handoff.json').read_text())
    expected_rounds = list(range(handoff['config']['budget'] + 1))
    if [int(row['iteration']) for row in rows] != expected_rounds:
        raise ValueError('missing, duplicate or out-of-order experiment row')
    if len({row['commit'] for row in rows}) != len(rows):
        raise ValueError('each experiment must name its own immutable revision')
    selected = rows[handoff['selected_iteration']]
    if selected['status'] != 'keep' or selected['commit'] != handoff['selected_trial_commit']:
        raise ValueError('selected experiment does not match the handoff')
    with tempfile.TemporaryDirectory(prefix='compass-replay-') as work:
        clone = Path(work) / 'trial'
        subprocess.run(['git', 'clone', '--quiet', '--local', str(REPO), str(clone)], check=True)
        subprocess.run(['git', '-C', str(clone), 'fetch', '--quiet', str(RUN / 'candidates.bundle'),
                        'refs/heads/codex/repo-structure:refs/heads/verified-trials'], check=True)
        ar = clone / 'research/autoresearch'
        cache = REPO / 'research/autoresearch/_panel_cache.npz'
        if cache.is_file():
            (ar / cache.name).symlink_to(cache)
        for row in rows:
            subprocess.run(['git', '-C', str(clone), 'checkout', '--quiet', '--detach', row['commit']], check=True)
            if json.loads((ar / 'candidate.json').read_text()) != json.loads(row['spec']):
                raise ValueError(f"iteration {row['iteration']}: spec differs from recorded revision")
            output = subprocess.check_output([sys.executable, str(ar / 'score_input_trial.py'),
                                              str(ar / 'candidate.json'), '--extra', str(args.extra.resolve())], text=True)
            actual = json.loads(output)
            for field, saved in [('metric', 'metric'), ('dev', 'mae_dev'), ('holdout', 'mae_holdout'), ('cv', 'level_ratio_cv')]:
                if actual['n'] != 64 or not math.isclose(actual[field], float(row[saved]), rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"iteration {row['iteration']}: mismatch for {saved}: {actual[field]}")
    print(f'{len(rows)} rows reproduced on all 64 dates (tolerance 1e-9).')


if __name__ == '__main__':
    main()
