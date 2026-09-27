#!/usr/bin/env python3
"""Exercise exported files with ffprobe; requires a built binary, ffmpeg and Vulkan.

Uses temporary local test media. No downloads, commercial ROMs or desktop session.
Agent/core checks report SKIP when their optional dependencies are unavailable.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def run(args, env=None, stdout_only=False):
    proc = subprocess.run([str(a) for a in args], env=env, capture_output=True, text=True, timeout=120)
    if proc.returncode:
        raise RuntimeError(f"{' '.join(map(str, args))}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout if stdout_only else proc.stdout + proc.stderr


def probe(path):
    return json.loads(run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', path], stdout_only=True))['streams']


def verify(path, frames, audio):
    streams = probe(path)
    v = next(s for s in streams if s['codec_type'] == 'video')
    assert int(v['nb_read_frames']) == frames, (path, v)
    sounds = [s for s in streams if s['codec_type'] == 'audio']
    assert bool(sounds) == audio, (path, streams)
    if path.suffix == '.webm' and audio:
        assert sounds[0]['codec_name'] == 'opus', sounds
    assert v.get('color_space') == 'bt709', v
    assert v.get('color_transfer') == 'iec61966-2-1', v
    print(f'PASS {path.name}: {frames} frames, audio={audio}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='target/debug/crtulum')
    opts = parser.parse_args()
    binary = Path(opts.binary).resolve()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    # Firejail-packaged media tools have private /tmp directories. Stage beside
    # the project so ffmpeg and ffprobe see the same files, as exports do.
    with tempfile.TemporaryDirectory(prefix='.crtulum-verify-', dir=root) as tmp:
        work = Path(tmp)
        # Startup/collapse used to feed inverse raster coordinates into defocus,
        # producing enormous convolution loops (and a hardware GPU reset).
        for phase, value in [('WARMUP', '0'), ('WARMUP', '0.25'), ('COLLAPSE', '1')]:
            shot = work / f'{phase}-{value}.png'
            run([binary, '--preset', 'rca', '--shot', shot, '160x120'],
                dict(env, **{f'CRTULUM_{phase}': value}))
            assert shot.stat().st_size > 100, shot
        print('PASS power warm-up/collapse rendering completes', flush=True)
        for preset in ['trinitron', 'panasonic', 'slotmask', 'rca', 'pvm', 'arcade',
                       'vga', 'diamondtron', 'green', 'amber']:
            shot = work / f'preset-{preset}.png'
            run([binary, '--preset', preset, '--shot', shot, '120x90'], env)
            assert shot.stat().st_size > 100, shot
        print('PASS all ten presets render', flush=True)
        # Connections must work on every tube, including monochrome/PC presets.
        # RF must be a distinct rendered path, not merely a label for composite.
        for preset in ['trinitron', 'panasonic', 'slotmask', 'rca', 'pvm', 'arcade',
                       'vga', 'diamondtron', 'green', 'amber']:
            composite, rf = work / f'{preset}-composite.png', work / f'{preset}-rf.png'
            for mode, shot in [('composite', composite), ('rf', rf)]:
                run([binary, '--preset', preset, '--input', mode, '--shot', shot, '120x90'], env)
            assert composite.read_bytes() != rf.read_bytes(), preset
        for mode in ['auto', 's-video', 'rgb', 'component']:
            shot = work / f'input-{mode}.png'
            run([binary, '--preset', 'pvm', '--input', mode, '--shot', shot, '120x90'], env)
        assert (work / 'input-auto.png').read_bytes() == (work / 'input-rgb.png').read_bytes()
        assert (work / 'input-rgb.png').read_bytes() == (work / 'input-component.png').read_bytes()
        assert (work / 'input-s-video.png').read_bytes() != (work / 'input-rgb.png').read_bytes()
        print('PASS connections on all presets; RF differs from composite; clean aliases agree', flush=True)
        for args in [['--preset', 'trinitrron'], ['--preset'], ['--input', 'composit']]:
            proc = subprocess.run([str(binary), *args], env=env, capture_output=True, text=True, timeout=10)
            assert proc.returncode == 2, (args, proc.stderr)
            assert '[gpu]' not in proc.stderr, proc.stderr
        print('PASS invalid preset/input rejected before GPU/window startup', flush=True)
        source = work / 'source.mp4'
        # A shorter audio stream must not truncate the half-second picture.
        run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=80x60:rate=24:duration=0.5',
             '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=0.1',
             '-c:v', 'libx264', '-c:a', 'aac', source])
        common = ['--size', '160x120', '--ssaa', '1']
        for codec, suffix in [('x264', '.mp4'), ('x265', '.mkv'), ('vp9', '.webm'), ('ffv1', '.mkv')]:
            out = work / (codec + suffix)
            run([binary, '--render', source, out, '--codec', codec, *common], env)
            verify(out, 12, True)
        out = work / 'trimmed.mp4'
        run([binary, '--render', source, out, '--start', '0.125', '--duration', '0.25', '--no-audio', *common], env)
        verify(out, 6, False)

        script = work / 'override.crts'
        script.write_text('source nonexistent.mp4\npreset rca\ninterlace on\nat 0.1 degauss\nat 0.2 power off\n')
        out = work / 'override.mp4'
        log = run([binary, '--render', source, '--out', out, '--script', script, '--preset', 'pvm', *common], env)
        assert 'tube    pvm' in log, log
        verify(out, 12, True)

        stills = work / 'stills'
        stills.mkdir()
        for i, (ext, color) in enumerate([('PNG', 'red'), ('jpg', 'green'), ('bmp', 'blue'), ('tga', 'white')]):
            run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'color={color}:size=80x60',
                 '-frames:v', '1', stills / f'{i}.{ext}'])
        out = work / 'stills.mp4'
        run([binary, '--render', stills, out, '--fps', '10', '--start', '0.1', '--duration', '0.2', *common], env)
        verify(out, 2, False)

        # Both spellings must render identical pixels, including multi-field
        # cadence and monochrome history, with no video/YUV round trip.
        clip_dir, render_dir = work / 'clip-png', work / 'render-png'
        png_options = ['--fps', '24', '--preset', 'green', *common]
        run([binary, '--clip', stills, clip_dir, *png_options], env)
        run([binary, '--render', stills, render_dir, '--codec', 'png', *png_options], env)
        clip_frames = sorted(clip_dir.glob('f_*.png'))
        render_frames = sorted(render_dir.glob('f_*.png'))
        assert len(clip_frames) == len(render_frames) == 4
        for clip_frame, render_frame in zip(clip_frames, render_frames):
            assert clip_frame.read_bytes() == render_frame.read_bytes(), (clip_frame, render_frame)
        print('PASS clip/render PNG output is identical', flush=True)

        # PNG exports must retain source audio as synchronized lossless PCM,
        # pad a short track, respect seeks, and honor --no-audio.
        for spelling in ['--clip', '--render']:
            sequence = work / f'audio-{spelling[2:]}'
            codec = ['--codec', 'png'] if spelling == '--render' else []
            run([binary, spelling, source, sequence, *codec, '--fps', '24',
                 '--start', '0.0416666667', '--duration', '0.25', *common], env)
            assert len(list(sequence.glob('f_*.png'))) == 6
            sound = probe(sequence / 'audio.wav')[0]
            assert sound['codec_name'] == 'pcm_f32le', sound
            assert abs(float(sound['duration']) - 0.25) < 1 / 48000, sound
            expected = subprocess.check_output([
                'ffmpeg', '-v', 'error', '-ss', '0.0416666667', '-i', str(source),
                '-map', '0:a:0', '-af', 'apad', '-t', '0.25', '-f', 'f32le', 'pipe:1'])
            actual = subprocess.check_output([
                'ffmpeg', '-v', 'error', '-i', str(sequence / 'audio.wav'), '-f', 'f32le', 'pipe:1'])
            assert actual == expected, 'PNG sidecar changed source samples or seek timing'
        silent = work / 'silent-png'
        run([binary, '--render', source, silent, '--codec', 'png', '--no-audio', *common], env)
        assert not (silent / 'audio.wav').exists()
        assert not (clip_dir / 'audio.wav').exists()
        print('PASS PNG audio, seek, duration, padding, and --no-audio', flush=True)

        # A native tall/odd PNG raster and odd output dimensions must not be
        # silently reduced to television lines or rounded for YUV subsampling.
        native_dir = work / 'native'
        native_dir.mkdir()
        run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=81x601',
             '-frames:v', '1', native_dir / '1.png'])
        native_out = work / 'native-out'
        settings = work / 'native.crts'
        settings.write_text('size 161x121\nfps 24\n')
        log = run([binary, '--clip', native_dir, native_out, '--script', settings, '--ssaa', '1'], env)
        assert 'signal 81x601' in log and '@ 161x121 24 fps' in log, log
        stream = probe(native_out / 'f_0001.png')[0]
        assert (stream['width'], stream['height']) == (161, 121), stream
        print('PASS native signal, odd PNG size, and clip script defaults', flush=True)

        sampled = work / 'sampled-clip'
        log = run([binary, '--clip', stills, sampled, *common], dict(env, CRTULUM_DT='0.05'))
        assert '20 fps' in log, log
        assert len(list(sampled.glob('f_*.png'))) == 4
        # An explicit script rate takes precedence over the diagnostic interval.
        log = run([binary, '--clip', native_dir, work / 'sampled-script', '--script', settings,
                   '--ssaa', '1'], dict(env, CRTULUM_DT='0.05'))
        assert '24 fps' in log, log
        print('PASS clip sampling interval and explicit fps precedence', flush=True)

        cores = Path.home() / '.config/retroarch/cores'
        core = next((cores / f'{n}_libretro.so' for n in ['nestopia', 'fceumm', 'quicknes']
                     if (cores / f'{n}_libretro.so').is_file()), None)
        if core:
            script.write_text('frames 30\nframe 1 hold a\nframe 8 release a\n')
            out = work / 'rom.webm'
            log = run([binary, '--render', out, '--rom', root / 'examples/inputtest.nes', '--core', core,
                       '--script', script, '--start', '0.1', '--duration', '0.1', *common],
                      dict(env, CRTULUM_DEBUG_INPUT='1'))
            assert 'frame     1' in log and 'frame     8' in log, log
            verify(out, 6, True)
            out = work / 'rom-png'
            log = run([binary, '--render', out, '--codec', 'png', '--rom', root / 'examples/inputtest.nes',
                 '--core', core, '--script', script, '--start', '0.1', '--duration', '0.1', *common], env)
            assert 'reusing cached recording' in log, log
            assert len(list(out.glob('f_*.png'))) == 6
            assert probe(out / 'audio.wav')[0]['codec_name'] == 'pcm_f32le'
            print('PASS ROM PNG audio', flush=True)
            # Camera/preset edits must reuse emulation; input edits must record
            # a new run. Cached frames retain precisely the same rendered pixels.
            cached = work / 'rom-cached'
            log = run([binary, '--render', cached, '--codec', 'png', '--rom', root / 'examples/inputtest.nes',
                       '--core', core, '--script', script, '--start', '0.1', '--duration', '0.1', *common], env)
            assert 'reusing cached recording' in log, log
            for a, b in zip(sorted(out.glob('f_*.png')), sorted(cached.glob('f_*.png'))):
                assert a.read_bytes() == b.read_bytes()
            script.write_text('frames 30\nframe 1 hold b\nframe 8 release b\npreset pvm\n')
            log = run([binary, '--render', work / 'rom-new-input', '--codec', 'png', '--rom', root / 'examples/inputtest.nes',
                       '--core', core, '--script', script, '--start', '0.1', '--duration', '0.1', *common], env)
            assert 'reusing cached recording' not in log and 'cached recording' in log, log
            print('PASS ROM cache reuse, identical frames, and input invalidation', flush=True)
        else:
            print('SKIP ROM export: no supported NES core installed', flush=True)

        agent = Path.home() / '.local/share/crtulum/agents/Merlin'
        if agent.is_dir():
            script.write_text(f'agent "{agent}"\nat 0 agent show\nat 0.05 agent say "Hi."\n')
            out = work / 'agent.webm'
            tts = 'ffmpeg -v error -f lavfi -i sine=frequency=880:sample_rate=22050 -t 0.05 -y {out}'
            run([binary, '--render', source, out, '--script', script, *common], dict(env, CRTULUM_TTS=tts))
            verify(out, 12, True)
            out = work / 'agent-png'
            run([binary, '--render', source, out, '--codec', 'png', '--script', script, *common],
                dict(env, CRTULUM_TTS=tts))
            assert len(list(out.glob('f_*.png'))) == 12
            assert probe(out / 'audio.wav')[0]['codec_name'] == 'pcm_f32le'
            print('PASS Agent PNG audio mix', flush=True)
        else:
            print('SKIP Agent export: Merlin artwork not installed', flush=True)

        bad = subprocess.run([str(binary), '--render', str(source), str(work / 'bad.mp4'), '--codec', 'typo'],
                             capture_output=True, text=True, env=env)
        assert bad.returncode != 0 and 'unknown codec' in bad.stderr, bad.stderr
        print('PASS invalid codec rejected', flush=True)


if __name__ == '__main__':
    main()
