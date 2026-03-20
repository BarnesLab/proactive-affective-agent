#!/usr/bin/env python3
"""Quality check for pilot_v2 record files.
Returns list of (version, user_id) pairs that need re-running.
Used by sonnet_watcher.sh to detect bad-quality files."""

import json
import os
import glob
import re
import sys

BASE = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'pilot_v2')
ARCHIVE = os.path.join(BASE, '_bad')

MIN_RECORDS = 10
MAX_EMPTY_RATIO = 0.05


def check_quality(filepath):
    """Check if a record file passes quality. Returns (ok, reason)."""
    try:
        with open(filepath) as f:
            lines = f.readlines()
        records = [json.loads(l) for l in lines if l.strip()]
    except Exception as e:
        return False, f"parse error: {e}"

    if len(records) < MIN_RECORDS:
        return False, f"too few records ({len(records)})"

    empty = sum(1 for r in records if not isinstance(r.get('prediction'), dict) or len(r.get('prediction', {})) == 0)
    if empty > len(records) * MAX_EMPTY_RATIO:
        return False, f"empty predictions ({empty}/{len(records)})"

    return True, "ok"


def scan_bad_files(versions=None, archive=False):
    """Scan all record files, return bad ones. Optionally archive them."""
    if versions is None:
        versions = ['callm', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    
    bad = []
    for f in sorted(glob.glob(os.path.join(BASE, '*_records.jsonl'))):
        basename = os.path.basename(f)
        # Skip CALLM uppercase (they're hardlinks to lowercase)
        if basename.startswith('CALLM_'):
            continue
        
        m = re.match(r'(callm|v[1-6])_user(\d+)_records\.jsonl', basename, re.IGNORECASE)
        if not m:
            continue
        
        ver = m.group(1).lower()
        uid = int(m.group(2))
        
        if ver not in versions:
            continue
        
        ok, reason = check_quality(f)
        if not ok:
            bad.append((ver, uid, reason, f))
            if archive:
                os.makedirs(ARCHIVE, exist_ok=True)
                dest = os.path.join(ARCHIVE, basename)
                # Also move uppercase CALLM variant if exists
                os.rename(f, dest)
                upper = f.replace(f'/{ver}_', f'/{ver.upper()}_')
                if os.path.exists(upper) and upper != f:
                    try:
                        os.rename(upper, os.path.join(ARCHIVE, os.path.basename(upper)))
                    except:
                        pass
    
    return bad


def _get_expected_entries():
    """Build mapping of user_id -> expected EMA entry count from test splits."""
    import pandas as pd
    from pathlib import Path
    splits_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'splits'))
    expected = {}
    if not splits_dir.exists():
        return expected
    for csv_file in sorted(splits_dir.glob('group_*_test.csv')):
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        if 'Study_ID' not in df.columns:
            continue
        for uid, count in df['Study_ID'].value_counts().items():
            expected[int(uid)] = int(count)
    return expected


def get_done_set(versions=None, skip_bad=False):
    """Return set of 'version_userXXX' keys that are truly complete.

    Completion is determined by checking checkpoint n_entries against the
    expected entry count from test splits (>= 95% = complete).
    Falls back to records-file existence only if checkpoint is missing.
    """
    if versions is None:
        versions = ['callm', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

    expected = _get_expected_entries()
    checkpoint_dir = os.path.join(BASE, 'checkpoints')
    done = set()

    for ver in versions:
        for cp_file in sorted(glob.glob(os.path.join(checkpoint_dir, f'{ver}_user*_checkpoint.json'))):
            basename = os.path.basename(cp_file)
            m = re.match(r'(callm|v[1-6])_user(\d+)_checkpoint\.json', basename, re.IGNORECASE)
            if not m:
                continue
            uid = int(m.group(2))
            try:
                with open(cp_file) as fh:
                    data = json.load(fh)
                n_entries = data.get('n_entries', len(data.get('predictions', [])))
            except Exception:
                continue

            exp = expected.get(uid, 0)
            if exp == 0:
                continue

            # Complete if checkpoint has >= 95% of expected entries
            if n_entries >= exp * 0.95:
                done.add(f'{ver}_user{uid}')

    return done


if __name__ == '__main__':
    if '--archive' in sys.argv:
        bad = scan_bad_files(archive=True)
        print(f"Archived {len(bad)} bad files to {ARCHIVE}/")
        for ver, uid, reason, f in bad:
            print(f"  {ver}_user{uid}: {reason}")
    elif '--done' in sys.argv:
        done = get_done_set()
        for key in sorted(done):
            print(key)
    else:
        bad = scan_bad_files()
        print(f"Bad quality files: {len(bad)}")
        for ver, uid, reason, f in bad:
            print(f"  {ver}_user{uid}: {reason}")
