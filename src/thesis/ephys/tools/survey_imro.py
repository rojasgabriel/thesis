"""Pick a V1 channel map (IMRO) from a SpikeGLX probe survey.

Question: which 384 sites of an NP2.0 4-shank probe record the most
flash-evoked V1 activity?

The SpikeGLX survey-map rate does not locate V1: sites outside the brain can
have a high rate, and exposed sites respond to the light itself. Thus this tool
measures the flash response of each site from the raw survey.

- Spikes: threshold crossings (-4.5 x MAD / 0.6745) after a 300-6000 Hz
  band-pass and a common median reference, separately for each survey segment
  (svySBTT; 1 s after settling is skipped).
- Alignment: first flash of each train (OneBox XA0 carrier > 10000, pulses
  merged when gaps < 20 ms, a new train after > 1 s), mapped to probe time by
  a linear fit of the SY edges.
- Response: rate 20-80 ms after the first flash - rate in the 200 ms before it,
  in spikes/s. The 20-80 ms window skips the 0-20 ms light artifact.
- Exclusions: samples with > 10% of the channels at the ADC limit (lick steps);
  flashes within 210 ms of such samples.
- Selection: one solid window of 96 sites per shank, inside --band (µm from
  the tip). The windows maximize the summed positive response with 384 unique
  channels. Responses outside the band (deeper structures, light artifact) are
  not used.

Run without --band first. It saves the per-site responses (.npz) and a figure
of response against depth, so that you can choose the V1 band (latency
20-40 ms). Run again with --band; the cached responses are used.

    uv run survey-imro GRB070.ap.bin --obx GRB070.obx.bin --out GRB070_jointips_pixelmap.imro
    uv run survey-imro GRB070.ap.bin --obx GRB070.obx.bin --out GRB070_jointips_pixelmap.imro --band 2880 4080

The IMRO uses joined tips and the (2013,384) header (SpikeGLX 20251218), the
same output as pixelmap (github.com/m-beau/pixelmap) for these electrodes.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

# SpikeGLX NP2.0 4-shank wiring: channel block (48 ch) -> electrode block per shank.
BLOCK = [
    [0, 6, 1, 7, 2, 4, 3, 5],
    [6, 0, 7, 1, 4, 2, 5, 3],
    [2, 4, 3, 5, 0, 6, 1, 7],
    [4, 2, 5, 3, 6, 0, 7, 1],
]
SITES = 96
BINS = np.arange(-200, 201, 10)  # ms


def electrode(shank: int, bank: int, ch: np.ndarray) -> np.ndarray:
    return 384 * bank + 48 * np.asarray(BLOCK[shank])[ch // 48] + ch % 48


def channel(shank: int, e: np.ndarray) -> np.ndarray:
    inv = np.argsort(BLOCK[shank])
    return 48 * inv[(e % 384) // 48] + e % 48


def _meta(path: Path) -> str:
    return path.with_suffix(".meta").read_text()


def _rising(x: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.diff(x.astype(np.int8)) == 1) + 1


def flash_responses(ap_path: Path, obx_path: Path) -> dict:
    meta = _meta(ap_path)
    fs = float(re.search(r"imSampRate=([\d.]+)", meta).group(1))
    fso = float(re.search(r"obSampRate=([\d.]+)", _meta(obx_path)).group(1))
    sbtt = [
        [int(float(v)) for v in g.split()]
        for g in re.findall(r"\(([\d. ]+)\)", re.search(r"svySBTT=(.*)", meta).group(1))
    ]
    ap = np.memmap(ap_path, np.int16, "r").reshape(-1, 385)
    ob = np.memmap(obx_path, np.int16, "r").reshape(-1, 4)
    n = ap.shape[0]

    # OneBox -> probe clock from the shared SY square wave.
    ti = _rising((np.asarray(ap[:, 384]) >> 6) & 1) / fs
    to = _rising(np.asarray(ob[:, 3]) > 0) / fso
    j = np.clip(np.searchsorted(ti, to), 1, len(ti) - 1)
    j = np.where(np.abs(ti[j - 1] - to) < np.abs(ti[j] - to), j - 1, j)
    ok = np.abs(ti[j] - to) < 0.5
    slope, icpt = np.polyfit(to[ok], ti[j[ok]], 1)
    resid = np.abs(ti[j[ok]] - (slope * to[ok] + icpt)).max() * 1e3
    print(f"sync: {ok.sum()} edges, max residual {resid:.2f} ms")

    c = _rising(np.asarray(ob[:, 0]) > 10000) / fso
    fl = slope * c[np.r_[True, np.diff(c) > 0.020]] + icpt
    first = np.r_[True, np.diff(fl) > 1.0]
    lim = int(np.abs(np.asarray(ap[::1000, :384])).max())
    print(f"{fl.size} flashes, {first.sum()} trains, ADC limit {lim}")

    sos = butter(3, [300, 6000], "bandpass", fs=fs, output="sos")
    chunk, pad = int(5 * fs), int(0.05 * fs)
    rows = []
    for k, (sh, bk, _, settled) in enumerate(sbtt):
        s0 = settled + int(fs)
        s1 = sbtt[k + 1][2] if k + 1 < len(sbtt) else n
        spikes = [[] for _ in range(384)]
        bad, thr = [], None
        for a0 in range(s0, s1, chunk):
            lo, hi = max(s0, a0 - pad), min(s1, a0 + chunk + pad)
            raw = np.asarray(ap[lo:hi, :384])
            x = sosfiltfilt(sos, raw.astype(np.float32), axis=0)
            x -= np.median(x, 1, keepdims=True)
            if thr is None:
                thr = -4.5 * np.median(np.abs(x), 0) / 0.6745
            k0, k1 = a0 - lo, min(a0 + chunk, s1) - lo
            railed = (np.abs(raw[k0:k1]) >= lim - 2).mean(1) > 0.1
            bad.append((np.flatnonzero(railed) + a0) / fs)
            t, ch = np.nonzero((x[k0 + 1 : k1] < thr) & (x[k0 : k1 - 1] >= thr))
            for chn in np.unique(ch):
                spikes[chn].append((t[ch == chn] + a0 + 1) / fs)
        bad = np.unique(np.round(np.concatenate(bad), 2))
        keep = (fl > s0 / fs + 0.2) & (fl < s1 / fs - 0.2) & first
        f1 = fl[keep]
        f1 = f1[np.searchsorted(bad, f1 - 0.21) == np.searchsorted(bad, f1 + 0.21)]
        for chn in range(384):
            st = np.concatenate(spikes[chn]) if spikes[chn] else np.array([])
            st = st[~np.isin(np.round(st, 2), bad)]
            h = sum(
                (np.histogram((st - f) * 1e3, BINS)[0] for f in f1),
                np.zeros(len(BINS) - 1),
            )
            rows.append((sh, electrode(sh, bk, np.array(chn)), len(f1), *h))
        print(
            f"shank {sh} bank {bk}: {len(f1)} first flashes, {bad.size / 100:.1f} s excluded",
            flush=True,
        )
    a = np.array(rows)
    h, nf = a[:, 3:], a[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        resp = h[:, 22:28].sum(1) / (nf * 0.06) - h[:, :20].sum(1) / (nf * 0.2)
    return dict(
        shank=a[:, 0].astype(int),
        electrode=a[:, 1].astype(int),
        n_first=nf,
        psth=h,
        response=np.nan_to_num(resp),
    )


def pick_windows(r: dict, band: tuple[float, float]) -> tuple[list[int], float]:
    """First electrode of the best 96-site window per shank (exhaustive, bounded)."""
    rp = np.zeros((4, 1280))
    rp[r["shank"], r["electrode"]] = np.clip(r["response"], 0, None)
    lo, hi = (int(v // 15 * 2) for v in band)
    cands = []
    for s in range(4):
        c = [
            (
                rp[s, e0 : e0 + SITES].sum(),
                e0,
                sum(1 << int(ch) for ch in channel(s, np.arange(e0, e0 + SITES))),
            )
            for e0 in range(lo, hi - SITES + 2, 2)
        ]
        if not c:
            raise ValueError(f"band {band} is shorter than {SITES} sites")
        cands.append(sorted(c, reverse=True))
    tail = [sum(c[0][0] for c in cands[s:]) for s in range(5)]  # upper bounds
    best = [-1.0, None]

    def search(s: int, used: int, total: float, chosen: list[int]) -> None:
        if s == 4:
            if total > best[0]:
                best[:] = [total, list(chosen)]
            return
        for score, e0, mask in cands[s]:
            if total + score + tail[s + 1] <= best[0]:
                return
            if not used & mask:
                search(s + 1, used | mask, total + score, chosen + [e0])

    search(0, 0, 0.0, [])
    if best[1] is None:
        raise ValueError("no window combination with 384 unique channels")
    return best[1], best[0]


def write_imro(path: Path, starts: list[int]) -> None:
    entries = []
    for s, e0 in enumerate(starts):
        e = np.arange(e0, e0 + SITES)
        entries += zip(channel(s, e), [s] * SITES, e // 384, e)
    entries.sort()
    assert [c for c, *_ in entries] == list(range(384)), "channels not unique"
    # Joined tips; channels 0-3 carry refids 2-5 (pixelmap 'Join Tips' convention).
    body = "".join(
        f"({c} {s} {b} {2 + c if c < 4 else 2} {e})" for c, s, b, e in entries
    )
    path.write_text(f"(2013,384){body}\n")


def plot(r: dict, path: Path, starts: list[int] | None) -> None:
    """Response against depth (mean of the 2 sites per row) for each shank.

    Neural V1 responds at 20-80 ms; light on exposed sites responds at 0-20 ms.
    """
    h, nf = r["psth"], r["n_first"]
    with np.errstate(invalid="ignore", divide="ignore"):
        base = h[:, :20].sum(1) / (nf * 0.2)
        early = np.nan_to_num(h[:, 20:22].sum(1) / (nf * 0.02) - base)
    traces = [
        ("20–80 ms (neural)", r["response"], "k"),
        ("0–20 ms (light)", early, "#E69F00"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10, 6), sharex=True, sharey=True)
    for s, ax in enumerate(axes):
        m = r["shank"] == s
        z = r["electrode"][m] // 2 * 15
        rows, inv = np.unique(z, return_inverse=True)
        for k, (label, v, color) in enumerate(traces):
            mean = np.bincount(inv, v[m]) / np.bincount(inv)
            ax.plot(mean, rows, color=color, lw=0.7)
            if s == 0:
                ax.text(
                    0.97,
                    0.72 - 0.04 * k,
                    label,
                    color=color,
                    fontsize=8,
                    ha="right",
                    transform=ax.transAxes,
                )
        ax.axvline(0, color="0.6", ls="--", lw=0.8)
        if starts is not None:
            lo, hi = starts[s] // 2 * 15, (starts[s] + SITES - 1) // 2 * 15
            ax.axhspan(lo, hi, color="#0072B2", alpha=0.2, lw=0)
            if s == 0:
                ax.text(
                    0.97,
                    0.64,
                    "new map",
                    color="#0072B2",
                    fontsize=8,
                    ha="right",
                    transform=ax.transAxes,
                )
        ax.text(-0.1, 1.02, "abcd"[s], transform=ax.transAxes, weight="bold")
        ax.set_title(f"shank {s}", fontsize=10)
    axes[0].set_ylabel("depth from tip (µm)")
    fig.supxlabel("first-flash response (spikes/s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    print(f"figure: {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ap", type=Path, help="survey .ap.bin (with .ap.meta)")
    p.add_argument(
        "--obx", type=Path, required=True, help="OneBox .obx.bin (with .obx.meta)"
    )
    p.add_argument("--out", type=Path, required=True, help="output .imro path")
    p.add_argument(
        "--band",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help="V1 band, µm from the tip",
    )
    a = p.parse_args()
    cache = a.out.with_suffix(".npz")
    if cache.exists():
        r = dict(np.load(cache))
    else:
        r = flash_responses(a.ap, a.obx)
        np.savez(cache, **r)
    starts = None
    if a.band:
        starts, total = pick_windows(r, tuple(a.band))
        write_imro(a.out, starts)
        for s, e0 in enumerate(starts):
            print(f"shank {s}: {e0 // 2 * 15}-{(e0 + SITES - 1) // 2 * 15} µm")
        print(f"summed response {total:.0f}; IMRO: {a.out}")
    plot(r, a.out.with_suffix(".svg"), starts)


if __name__ == "__main__":
    main()
