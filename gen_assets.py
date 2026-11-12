#!/usr/bin/env python3
"""Generates the HD pack (textures, normal maps, music, sound effects) for Backrooms Escape.
Usage: python3 tools/gen_assets.py [out_dir] [music_seconds]
Needs: numpy, scipy, pillow."""
import os, sys, json, wave
import numpy as np
from scipy import signal
from PIL import Image, ImageDraw

OUT = sys.argv[1] if len(sys.argv) > 1 else 'app/src/main/assets/hd'
MUSIC_SECS = int(sys.argv[2]) if len(sys.argv) > 2 else 90
os.makedirs(OUT + '/tex', exist_ok=True); os.makedirs(OUT + '/audio', exist_ok=True)
rng = np.random.default_rng(20240607)
N = 1024
KY = (np.fft.fftfreq(N)[:, None]).astype(np.float32)
KX = (np.fft.rfftfreq(N)[None, :]).astype(np.float32)
Y, X = np.mgrid[0:N, 0:N]

# ---------------------------------------------------------------- image helpers
def col(h): return np.array([int(h[i:i+2], 16) for i in (1, 3, 5)], np.float32) / 255
def blob(sx, sy=None):
    sy = sx if sy is None else sy
    f = np.fft.rfft2(rng.standard_normal((N, N)).astype(np.float32))
    f *= np.exp(-2 * np.pi ** 2 * ((KX * sx) ** 2 + (KY * sy) ** 2))
    a = np.fft.irfft2(f, s=(N, N)); a /= a.std() + 1e-9
    return np.clip(a / 3, -1, 1).astype(np.float32)
def fractal(beta=1.5, lo=1.0, hi=None):
    k = np.hypot(KX * N, KY * N); k[0, 0] = 1
    f = np.fft.rfft2(rng.standard_normal((N, N)).astype(np.float32)) / k ** beta
    f[k < lo] = 0
    if hi: f[k > hi] = 0
    f[0, 0] = 0
    a = np.fft.irfft2(f, s=(N, N)); a /= a.std() + 1e-9
    return np.clip(a / 3, -1, 1).astype(np.float32)
def mix(m, a, b): return b * m[..., None] + a * (1 - m[..., None])
def grain(img, s=.025):
    return img + rng.standard_normal((N, N, 1)).astype(np.float32) * s + rng.standard_normal(img.shape).astype(np.float32) * s * .4
def smooth(t, a, b): t = np.clip((t - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
def lines(n, steps=14, step=60, w=3):
    im = Image.new('L', (N, N), 0); d = ImageDraw.Draw(im)
    for _ in range(n):
        x, y = rng.uniform(0, N), rng.uniform(0, N); a = rng.uniform(0, 6.28); pts = [(x, y)]
        for _ in range(steps):
            a += rng.normal(0, .5); x += np.cos(a) * step * rng.uniform(.3, 1); y += np.sin(a) * step * rng.uniform(.3, 1); pts.append((x, y))
        d.line(pts, fill=255, width=w)
    return np.asarray(im, np.float32) / 255
def save_img(path, a):
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(path, compress_level=6)
def save_normal(path, h, strength, size=512):
    if h.shape[0] != size: r = h.shape[0] // size; h = h.reshape(size, r, size, r).mean((1, 3))
    gx = np.roll(h, -1, 1) - np.roll(h, 1, 1); gy = np.roll(h, -1, 0) - np.roll(h, 1, 0)
    nx, ny, nz = -gx * strength, gy * strength, np.ones_like(h); l = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    save_img(path, np.stack([nx / l, ny / l, nz / l], -1) * .5 + .5)

# ---------------------------------------------------------------- walls / floors  (return color, height)
def wall0():
    s = (X % 128) < 48; hl = ((X % 128) >= 48) & ((X % 128) < 56)
    c = np.where(s[..., None], col('#b19d3a'), col('#c6b24c')); c = np.where(hl[..., None], col('#d4c15e'), c)
    dots = (((Y % 112) < 12) & (((X + 24) % 128) < 12)); c = np.where(dots[..., None], col('#a89436'), c)
    c = c * (1 + .10 * fractal(1.2, 2, 60)[..., None])
    c *= (1 - .45 * smooth(Y, N - 330, N))[..., None]
    st = smooth(blob(24, 90) * .5 + .5, .55, .8) * smooth(-Y, -420, 0); c *= (1 - .45 * st)[..., None]
    wood = 1 + .2 * blob(40, 1.2); bb = Y >= N - 90
    c = np.where(bb[..., None], col('#3d2e16') * wood[..., None], c); c = np.where(((Y >= N - 90) & (Y < N - 78))[..., None], col('#6a5230'), c)
    h = .5 + .05 * fractal(1.2, 2, 80) + .03 * s + .35 * bb
    return grain(c), h
def floor0():
    f = blob(.7, 1.6); g = fractal(1.3, 2, 120)
    c = col('#6a5d2b') * (1 + .16 * f[..., None] + .10 * g[..., None]); c *= (1 - .3 * smooth(blob(30) * .5 + .5, .6, .8))[..., None]
    return grain(c, .03), .5 + .3 * f + .1 * g
def wall1():
    c = col('#7b7b76') * (1 + .16 * fractal(1.3, 2, 100)[..., None])
    pits = smooth(fractal(.4, 40, 220) * .5 + .5, .72, .8); c *= (1 - .35 * pits)[..., None]
    seam = ((np.abs(X - 512) < 5) | (np.abs(Y - 512) < 5)).astype(np.float32); c *= (1 - .5 * seam)[..., None]
    cr = lines(5, w=3); c *= (1 - .55 * cr)[..., None]
    rust = smooth(blob(3, 70) * .5 + .5, .6, .85) * smooth(Y, 0, 300); c = mix(rust * .6, c, col('#6e3d1c') * (1 + .2 * blob(2)[..., None]))
    band = (Y >= N - 280) & (Y < N - 190); hz = ((X + Y) // 80) % 2 == 0
    c = np.where(band[..., None], np.where(hz[..., None], col('#d6b12a'), col('#1d1d1d')), c); c *= (1 - .4 * smooth(Y, N - 170, N))[..., None]
    return grain(c), .5 + .25 * fractal(1.4, 2, 120) - .3 * seam - .3 * cr - .25 * pits
def floor1():
    c = col('#505050') * (1 + .16 * fractal(1.2, 2, 150)[..., None])
    oil = smooth(blob(25) * .5 + .5, .62, .85); c *= (1 - .55 * oil)[..., None]
    cr = lines(4, w=2); c *= (1 - .6 * cr)[..., None]; seam = ((X < 4) | (Y < 4) | (np.abs(X - 512) < 3) | (np.abs(Y - 512) < 3)); c *= (1 - .5 * seam)[..., None]
    return grain(c, .04), .5 + .25 * fractal(1.3, 4, 250) - .3 * cr - .3 * seam
def wall2():
    up = col('#bcc3c0') * (1 + .04 * blob(8)[..., None]); weave = ((X // 4 + Y // 4) % 2).astype(np.float32)
    lo = col('#5f7783') * (1 + .10 * weave[..., None] - .05 + .05 * blob(2)[..., None])
    c = np.where((Y < 590)[..., None], up, lo); c = np.where(((Y >= 584) & (Y < 606))[..., None], col('#e8ecea'), c)
    c *= (1 - .25 * ((X % 256) < 3)[..., None] * (Y >= 606)[..., None]); c = np.where((Y >= N - 56)[..., None], col('#2f3a40'), c)
    c *= (1 - .35 * smooth(blob(40) * .5 + .5, .62, .85))[..., None]
    return grain(c), .5 + .04 * weave * (Y >= 606) + .2 * ((Y >= 584) & (Y < 606)) + .3 * (Y >= N - 56)
def floor2():
    w = ((X // 2 + Y // 2) % 2).astype(np.float32); f = blob(.8, 1.3)
    c = col('#3a464e') * (1 + .12 * f[..., None] + .08 * w[..., None]); c *= (1 - .5 * (((X % 256) < 3) | ((Y % 256) < 3)))[..., None]
    c *= (1 - .3 * smooth(blob(35) * .5 + .5, .6, .8))[..., None]
    return grain(c, .03), .5 + .15 * w + .2 * f
def wall3():
    tone = np.kron(rng.random((16, 16)).astype(np.float32), np.ones((64, 64), np.float32))
    inside = ((X % 64) > 2) & ((X % 64) < 62) & ((Y % 64) > 2) & ((Y % 64) < 62)
    c = np.where(inside[..., None], col('#c6e6e8') * (.94 + .1 * tone[..., None]), col('#8fa9ac'))
    c *= (1 - .3 * smooth(blob(30) * .5 + .5, .6, .85))[..., None] * np.array([.9, 1, .9], np.float32)
    w = smooth(Y, N - 420, N); c = mix(w * .6, c, col('#14646e')); c = np.where((Y >= N - 22)[..., None], col('#146e78'), c)
    return grain(c, .018), .5 + .25 * inside - .05 * tone
def floor3():
    inside = ((X % 128) > 3) & ((Y % 128) > 3); c = np.where(inside[..., None], col('#2a8790'), col('#0e4a52')) * (1 + .1 * blob(6)[..., None])
    a = fractal(1.1, 3, 40); caustic = np.exp(-(a * 4.5) ** 2); c = c + caustic[..., None] * np.array([.45, .6, .6], np.float32) * .55
    return grain(c, .02), .5 + .2 * inside
def wall4():
    xo = ((X + (Y // 128 % 2) * 64) % 128) - 64; v = (Y % 128) - 64; r = np.hypot(xo, v)
    c = np.tile(col('#5c1515'), (N, N, 1)); c = np.where((r < 40)[..., None], col('#470e0e'), c); c = np.where((r < 16)[..., None], col('#702020'), c)
    c = np.where(((r > 40) & (r < 46))[..., None], col('#6a1c1c'), c); c *= (1 + .10 * fractal(1.3, 2, 80)[..., None])
    c *= (1 - .5 * smooth(blob(35) * .5 + .5, .6, .85))[..., None]
    peel = smooth(blob(18) * .5 + .5, .72, .78) * smooth(-Y, -500, 0); c = mix(peel, c, col('#8a5a40'))
    bb = Y >= N - 104; c = np.where(bb[..., None], col('#1d0d0a') * (1 + .25 * blob(40, 1.2)[..., None]), c)
    c = np.where(((Y >= N - 104) & (Y < N - 92))[..., None], col('#3a1a12'), c)
    return grain(c, .03), .5 + .08 * (r < 40) + .05 * fractal(1.3, 2, 100) + .3 * bb - .1 * peel
def floor4():
    d = (np.abs(((X + Y) % 256) - 128) < 3) | (np.abs(((X - Y) % 256) - 128) < 3); f = blob(.8, 1.4)
    c = col('#3a0e0e') * (1 + .16 * f[..., None]); c = np.where(d[..., None], col('#a03c28') * .45, c)
    c *= (1 - .5 * smooth(blob(30) * .5 + .5, .62, .85))[..., None]
    return grain(c, .03), .5 + .15 * d + .2 * f
def ceil_tex(cb, cl, lt, li):
    n = 512; global N
    y, x = np.mgrid[0:n, 0:n]
    c = col(cb) * (1 + .06 * rng.standard_normal((n, n, 1)).astype(np.float32))
    holes = rng.random((n, n)) < .012; c[holes] *= .5
    seam = (x < 3) | (y < 3) | (np.abs(x - 252) < 2) | (np.abs(y - 252) < 2) | (np.abs(x - 256) < 2) | (np.abs(y - 256) < 2); c[seam] = col(cl)
    glow = np.clip(1 - np.hypot((x - 256) / 230., (y - 252) / 90.), 0, 1)[..., None]
    c = c + glow * col(lt) * .3
    pn = (x > 88) & (x < 424) & (y > 208) & (y < 304); c[pn] = col(lt)
    pi = (x > 104) & (x < 408) & (y > 220) & (y < 292); c[pi] = col(li)
    return np.clip(c + rng.standard_normal((n, n, 1)).astype(np.float32) * .01, 0, 1)

THEMES = [(wall0, floor0, 8, ('#bfb26a', '#8c8040', '#fff0b0', '#ffffff')), (wall1, floor1, 6, ('#6e6e6a', '#4a4a48', '#9ab4d8', '#e6f0ff')),
          (wall2, floor2, 5, ('#c9cfcc', '#8a908e', '#dfe8ec', '#ffffff')), (wall3, floor3, 4, ('#d8f0f0', '#9bbcbc', '#e0ffff', '#ffffff')),
          (wall4, floor4, 6, ('#2a0c0c', '#150606', '#ff7a5a', '#ffd0c0'))]
for i, (wf, ff, ns, cc) in enumerate(THEMES):
    c, h = wf(); save_img(f'{OUT}/tex/t{i}_wall.png', c); save_normal(f'{OUT}/tex/t{i}_walln.png', h, ns * 6)
    c, h = ff(); save_img(f'{OUT}/tex/t{i}_floor.png', c); save_normal(f'{OUT}/tex/t{i}_floorn.png', h, ns * 5)
    save_img(f'{OUT}/tex/t{i}_ceil.png', ceil_tex(*cc)); print('textures', i, flush=True)

# ---------------------------------------------------------------- audio helpers
SR = 22050
T = lambda s: np.arange(int(SR * s)) / SR
def lp(x, fc): return signal.sosfilt(signal.butter(2, min(fc, SR / 2 - 100) / (SR / 2), 'low', output='sos'), x)
def hp(x, fc): return signal.sosfilt(signal.butter(2, fc / (SR / 2), 'high', output='sos'), x)
def bp(x, lo, hi): return signal.sosfilt(signal.butter(2, [lo / (SR / 2), min(hi, SR / 2 - 100) / (SR / 2)], 'band', output='sos'), x)
def noise(s): return rng.standard_normal(int(SR * s))
def ex(s, tau): return np.exp(-T(s) / tau)
def norm(x, peak=.85): return x / (np.abs(x).max() + 1e-9) * peak
def reverb(x, rt=2.4, mixv=.35):
    n = int(SR * rt); ir = lp(rng.standard_normal(n) * np.exp(-np.arange(n) / SR / (rt / 4)), 3500)
    y = signal.fftconvolve(x, ir)[:len(x)]; y *= np.sqrt((x ** 2).mean() / ((y ** 2).mean() + 1e-12))
    return x * (1 - mixv) + y * mixv
def add(x, y, t0):
    i = int(t0 * SR)
    if i < len(x): m = min(len(y), len(x) - i); x[i:i + m] += y[:m]
def write_wav(path, x):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())
def ping(f, d=3):
    t = T(d); return sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / (d / 4) * (1 + k * .6)) for k, (r, a) in enumerate([(1, 1), (2.76, .4), (5.4, .2)])) * .5
def drip():
    t = T(.5); f = rng.uniform(900, 2200); return np.sin(2 * np.pi * (f + 900 * np.exp(-t * 40)) * t) * np.exp(-t / .06)
def thump(f=55, d=.6):
    t = T(d); return (np.sin(2 * np.pi * f * t * (1 + .6 * np.exp(-t * 12))) * np.exp(-t / (d / 4)) + .2 * lp(noise(d), 200) * np.exp(-t / .1))
def creak(d=1.4):
    t = T(d); f = 90 + 70 * np.sin(2 * np.pi * rng.uniform(.5, 1.5) * t) + 30 * t; ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph + 3 * np.sin(ph * 2.01)) * (np.sin(np.pi * t / d) ** 1.5); return bp(x, 150, 1800) * 1.4
def whisper(d=1.6):
    t = T(d); n = noise(d); x = bp(n, 700, 1100) + .8 * bp(n, 1800, 2600) + .5 * bp(n, 3200, 4200)
    return x * (np.sin(np.pi * t / d) ** 2) * (.6 + .4 * np.sin(2 * np.pi * rng.uniform(4, 8) * t))
def click(): return hp(noise(.05), 1500) * ex(.05, .008)
def heart(): return thump(48, .45) + np.concatenate([np.zeros(int(SR * .22)), thump(40, .4) * .7])[:int(SR * .45)]

def music(k, secs):
    n = int(SR * (secs + 4)); t = np.arange(n) / SR; x = np.zeros(n)
    P = {1: dict(f=[55, 110, 164.8, 220, 329.6], hum=[120, 240, 360, 480], nz=900, ev=['ping', 'creak']),
         2: dict(f=[41.2, 82.4, 123.5, 185], hum=[], nz=1500, ev=['drip', 'thump', 'ping']),
         3: dict(f=[65.4, 98, 130.8, 196], hum=[100, 200, 300], nz=2500, ev=['click', 'thump']),
         4: dict(f=[49, 73.4, 98, 147], hum=[], nz=4000, ev=['drip', 'ping']),
         5: dict(f=[46.2, 65.4, 92.5, 130.8], hum=[], nz=700, ev=['heart', 'whisper', 'creak'])}[k]
    for i, f in enumerate(P['f']):
        lfo = 1 + .35 * np.sin(2 * np.pi * rng.uniform(.03, .11) * t + rng.uniform(0, 6))
        vib = 1 + .0012 * np.sin(2 * np.pi * rng.uniform(.15, .3) * t)
        x += (.5 / (i + 1)) * lfo * np.sin(2 * np.pi * f * np.cumsum(vib) / SR + rng.uniform(0, 6))
    for f in P['hum']: x += .025 * np.sin(2 * np.pi * f * t) * (1 + .3 * np.sin(2 * np.pi * rng.uniform(.5, 3) * t))
    pad = lp(rng.standard_normal(n), P['nz']); x += .22 * pad / pad.std() * .35 * (.55 + .45 * np.sin(2 * np.pi * .06 * t + k))
    ntime = int(secs * rng.uniform(.25, .4))
    for _ in range(ntime):
        e = rng.choice(P['ev']); t0 = rng.uniform(0, secs)
        y = {'ping': lambda: ping(rng.choice([220, 261.6, 329.6, 392, 440]) * (1 if k != 5 else .75)) * .25, 'drip': lambda: drip() * .3,
             'thump': lambda: thump(rng.uniform(45, 70)) * .4, 'click': lambda: click() * .3, 'creak': lambda: creak() * .22,
             'heart': lambda: heart() * .5, 'whisper': lambda: whisper() * .12}[e]()
        add(x, y, t0)
    if k == 5:
        for b in np.arange(0, secs, 1.15): add(x, heart() * .35, b)
    x = reverb(x, 2.6, .38); x = norm(x)
    X_ = int(SR * 4); y = x[:n - X_].copy(); w = np.linspace(0, 1, X_)
    y[:X_] = x[:X_] * w + x[n - X_:] * (1 - w)          # seamless loop
    return norm(lp(y, 9000), .8)

for k in range(1, 6):
    write_wav(f'{OUT}/audio/music{k}.wav', music(k, MUSIC_SECS)); print('music', k, flush=True)

def step(theme):
    k = rng.uniform(.8, 1.2)
    if theme == 1:   # concrete: crisp tap + slap-back
        x = hp(noise(.3), 700) * ex(.3, .012 * k) + .5 * lp(noise(.3), 300) * ex(.3, .04) + .4 * np.sin(2 * np.pi * 95 * k * T(.3)) * ex(.3, .03)
        return reverb(x, .5, .25)
    if theme == 3:   # pool: splash + bubble
        t = T(.35); sp = bp(noise(.35), 500, 4000) * ex(.35, .06); bub = np.sin(2 * np.pi * (450 + 900 * t * 3) * k * t) * ex(.35, .05) * .4
        return sp + bub
    f = {0: 500, 2: 380, 4: 320}[theme]                   # carpets
    return lp(noise(.3), f * k) * ex(.3, .035) * 1.5 + .6 * np.sin(2 * np.pi * 70 * k * T(.3)) * ex(.3, .05)
def knock():
    x = np.zeros(int(SR * 1.2))
    for t in (0, .32 + rng.uniform(0, .1), .7 + rng.uniform(0, .15)):
        y = thump(rng.uniform(70, 110), .35); c = click() * .5; y[:len(c)] += c; add(x, y, t)
    return x
def stinger(d=1.2):
    t = T(d); f = 110 * (1 + 4 * t); ph = 2 * np.pi * np.cumsum(f) / SR
    return (signal.sawtooth(ph) * .4 + signal.sawtooth(ph * 1.414) * .3 + hp(noise(d), 900) * .5) * np.exp(-t / .5) * np.minimum(1, t * 60)
def growl(d=1.4):
    t = T(d); f = 60 + 25 * np.sin(2 * np.pi * 3 * t); ph = 2 * np.pi * np.cumsum(f) / SR
    return lp(signal.sawtooth(ph) * .6 + .4 * noise(d), 900) * (.6 + .4 * np.sign(np.sin(2 * np.pi * 28 * t))) * (np.sin(np.pi * t / d) ** .7) * 1.3
def gulp():
    x = np.zeros(int(SR * .9)); [add(x, np.sin(2 * np.pi * (180 + 160 * T(.18) * 5) * T(.18)) * ex(.18, .06), t) for t in (0, .25, .5)]; return x
def pop(): return hp(noise(.3), 1200) * ex(.3, .02) + .5 * np.sin(2 * np.pi * 900 * T(.3)) * ex(.3, .04)
def breath(d=1.8): t = T(d); return lp(noise(d), 1400) * (np.sin(np.pi * t / d) ** 2) * .7 + bp(noise(d), 2500, 3500) * (np.sin(np.pi * t / d) ** 3) * .15

man = {'v': 1, 'music': 5}
def wr(name, x, peak=.8): write_wav(f'{OUT}/audio/{name}.wav', norm(x, peak))
for th in range(5):
    for k in range(5): wr(f'step{th}_{k}', step(th), .55)
for nm, fn, c in [('creak', creak, 4), ('knock', knock, 3), ('whisper', whisper, 4), ('growl', growl, 3), ('stinger', stinger, 2), ('breath', breath, 2)]:
    for k in range(c): wr(f'{nm}{k}', fn(), .8)
wr('pop0', pop()); wr('gulp0', gulp())
json.dump(man, open(OUT + '/manifest.json', 'w'))
tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs)
print('TOTAL MB: %.1f' % (tot / 1048576))
