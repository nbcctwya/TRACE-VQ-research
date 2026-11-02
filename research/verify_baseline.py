"""Verify the original TRACE-VQ against the independent copy's snapshot."""

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Hash every original file, including large binary files.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'research/baseline'
    state = json.loads((evidence / 'source-state.json').read_text())
    manifest = json.loads((evidence / 'source-manifest.json').read_text())
    source = Path(state['source'])
    differences = []
    hashed = 0
    for relative, expected in manifest.items():
        path = source / relative
        try:
            actual = path.lstat()
        except FileNotFoundError:
            differences.append(f'Missing: {relative}')
            continue
        if expected['type'] == 'symlink':
            if not stat.S_ISLNK(actual.st_mode) or os.readlink(path) != expected['target']:
                differences.append(f'Symlink changed: {relative}')
        elif not stat.S_ISREG(actual.st_mode):
            differences.append(f'File type changed: {relative}')
        else:
            if (actual.st_size, actual.st_mtime_ns, stat.S_IMODE(actual.st_mode)) != (
                expected['size'], expected['mtime_ns'], expected['mode']
            ):
                differences.append(f'Metadata changed: {relative}')
            if args.full or expected['size'] <= 8 * 1024 * 1024:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        digest.update(chunk)
                hashed += 1
                if digest.hexdigest() != expected['sha256']:
                    differences.append(f'Content changed: {relative}')
    present = set()
    for current, dirs, files in os.walk(source, followlinks=False):
        for name in list(dirs):
            if (Path(current) / name).is_symlink():
                dirs.remove(name)
                files.append(name)
        present.update((Path(current) / name).relative_to(source).as_posix() for name in files)
    differences.extend(f'New file: {path}' for path in sorted(present - manifest.keys()))
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    for arguments, expected in (
        (['rev-parse', 'HEAD'], state['head'] + '\n'),
        (['branch', '--show-current'], state['branch'] + '\n'),
        (['status', '--porcelain=v1', '--untracked-files=all'], state['status_porcelain']),
    ):
        actual = subprocess.check_output(
            ['git', '--no-optional-locks', '-C', str(source), *arguments], env=env, text=True
        )
        if actual != expected:
            differences.append(f'Git state changed: {arguments[0]}')
    report = {
        'source': str(source), 'full_hash_check': args.full,
        'files_and_symlinks_checked': len(manifest), 'regular_files_hashed': hashed,
        'source_unchanged': not differences, 'differences': differences,
    }
    logs = root / 'research/logs'
    logs.mkdir(exist_ok=True)
    (logs / 'source-integrity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(differences))


if __name__ == '__main__':
    main()
