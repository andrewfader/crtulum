#!/usr/bin/env python3
"""Verify native and gamescope Vulkan presentation on the active desktop.

Requires a physical Vulkan GPU, gamescope + its WSI layer, and the Vulkan
validation layer. The nested HDR test tone maps to SDR; no display settings change.
"""
import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile


def check(binary, label, args, hdr=False):
    env = os.environ.copy()
    env['VK_INSTANCE_LAYERS'] = 'VK_LAYER_KHRONOS_validation'
    # File output avoids waiting for pipe EOF from gamescope helper processes.
    with tempfile.TemporaryFile(mode='w+') as log:
        proc = subprocess.Popen([str(binary), *args, '--verify-frames', '60'],
                                env=env, stdout=log, stderr=log, start_new_session=True)
        try:
            code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise RuntimeError(f'{label}: timed out')
        log.seek(0)
        output = log.read()
    completed = re.search(r'\[verify\] presented 60 frames; (.+)', output)
    errors = re.search(r'Validation Error|VUID-|panicked at', output)
    if code or not completed or errors or (hdr and not completed[1].startswith('HDR ')):
        raise RuntimeError(f'{label}: failed (exit {code})\n{output}')
    if not re.search(r'\[gpu\].*Vulkan', output):
        raise RuntimeError(f'{label}: no Vulkan adapter reported\n{output}')
    print(f'PASS {label}: Vulkan, {completed[1]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='target/debug/crtulum')
    opts = parser.parse_args()
    binary = Path(opts.binary).resolve()
    check(binary, 'native', [])
    check(binary, 'gamescope desktop', ['--gamescope'])
    check(binary, 'gamescope HDR → SDR', ['--gamescope-hdr-test', '--require-hdr'], hdr=True)


if __name__ == '__main__':
    main()
