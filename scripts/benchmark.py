#!/usr/bin/env python3
"""Compare two benchmark-enabled builds on one physical GPU, sequentially.

Reports GPU pass medians separately from synchronized wall time (not live FPS).
Alternates build order and requires identical final HDR pixels on every run.
Both binaries must implement --benchmark and its companion options.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--size', default='1920x1080')
    parser.add_argument('--source', default='320x240')
    parser.add_argument('--presets', nargs='+', default=['trinitron'])
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    opts = parser.parse_args()
    if opts.frames <= 0 or opts.repeats <= 0:
        parser.error('frames and repeats must be positive')
    binaries = [opts.baseline.resolve(), opts.candidate.resolve()]
    report = {'baseline': str(binaries[0]), 'candidate': str(binaries[1]), 'cases': []}
    with tempfile.TemporaryDirectory(prefix='crtulum-bench-') as temp:
        for preset in opts.presets:
            runs = [[], []]
            for repeat in range(opts.repeats):
                images = [Path(temp) / f'{i}.rgba16' for i in range(2)]
                for i in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                    proc = subprocess.run([str(binaries[i]), '--benchmark', opts.size,
                        '--benchmark-source', opts.source, '--preset', preset,
                        '--benchmark-frames', str(opts.frames), '--benchmark-output', str(images[i])],
                        capture_output=True, text=True, timeout=300)
                    if proc.returncode:
                        raise RuntimeError(f'{binaries[i]} failed:\n{proc.stderr}')
                    runs[i].append(json.loads(proc.stdout))
                if images[0].read_bytes() != images[1].read_bytes():
                    raise RuntimeError(f'{preset}: HDR pixels differ on repetition {repeat + 1}')
                print(f'{preset}: pair {repeat + 1}/{opts.repeats}, identical HDR pixels', flush=True)
            case = {'preset': preset, 'runs': runs, 'identical_hdr_pixels': True, 'timings': {}}
            for stage in ['accum_ms', 'tube_ms', 'wall_ms']:
                before, after = [statistics.median(r[stage]['median'] for r in build) for build in runs]
                case['timings'][stage] = {'before': before, 'after': after,
                    'reduction_percent': 100 * (1 - after / before)}
            report['cases'].append(case)
            opts.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(case['timings']), flush=True)


if __name__ == '__main__':
    main()
