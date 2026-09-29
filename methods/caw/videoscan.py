"""Answer-free input checks for the methods/caw pre-registration (CPU only, no model, no scoring).

    uv run --project envs/caw python -m methods.caw.videoscan --split val --compare-frames
    uv run --project envs/caw python -m methods.caw.videoscan --split test
    uv run --project envs/caw python -m methods.caw.videoscan --split val --github-csv <path to GitHub quantiphy_validation.csv>

1. Video scan (every video of the split, one at a time): does the file exist, would the authors'
   resolver (<video-dir>/<video_id>.mp4) find it, decord's frame count vs PyAV's demuxed packet
   count, container fps vs the table fps, and what the 16-frame sampling does with the clip
   length: fine (>= 16 frames), the authors' retry works, or their retry raises again (4k+3
   frames; the port's second retry is used). With --compare-frames, decord's frames at the
   sampled indices (get_batch, as qwen-vl-utils reads them) are compared with decord's own linear
   decode (a seek error would show here) and with PyAV's frames at the same indices (max |diff|;
   different FFmpeg builds may round the MPEG-4 Part 2 IDCT differently by a level or two).
   Output: results/caw_<split>_video_scan.json.
2. --github-csv (validation): the authors' README runs on quantiphy_validation.csv from GitHub
   Paulineli/QuantiPhy, not on the Hugging Face validation_dataset.csv used here. Only the id
   column and INPUT_COLUMNS are read from both files (pandas usecols): the answer column
   (ground_truth_posterior) is never parsed. Output: results/caw_val_github_csv_diff.json.

The loaders' question tables come from methods.caw.run.questions, which drops validation answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantiphy.data import ROOT, VAL_CSV

from methods.caw import recipe as R

INPUT_COLUMNS = ("video_id", "question", "ground_truth_prior", "depth_info", "fps", "video_type", "inference_type")


def sampling_plan(n: int) -> dict:
    """What qwen-vl-utils (nframes=16, rounded half-to-even to a multiple of 2) and the authors'
    retry do with a clip of n frames; the port's second retry uses the largest even count <= n."""
    if n >= R.VIDEO_NFRAMES:
        return {"case": "ok", "nframes": R.VIDEO_NFRAMES}
    if n < 2:
        return {"case": "too_short_for_any_retry", "nframes": None}
    authors_retry = round(n / 2) * 2  # round_by_factor(n, 2), Python round = half to even
    if authors_retry <= n:
        return {"case": "authors_retry", "nframes": authors_retry}
    return {"case": "authors_retry_raises_port_deviation", "nframes": n // 2 * 2, "authors_retry_asks": authors_retry}


def sample_indices(n: int, k: int) -> list[int]:
    """qwen-vl-utils' decord reader: linspace(0, n - 1, k).round()."""
    return np.linspace(0, n - 1, k).round().astype(int).tolist()


def _compare_frames(path: str, idx: list[int]) -> dict:
    """decord's frames at idx (get_batch, which seeks, as qwen-vl-utils reads them) vs decord's own
    linear decode (vr.next()) and vs PyAV's linear decode at the same indices."""
    import decord

    vr = decord.VideoReader(path, num_threads=1)
    d = vr.get_batch(idx).asnumpy()
    del vr
    want, lin = set(idx), {}
    vr = decord.VideoReader(path, num_threads=1)
    for i in range(max(idx) + 1):
        f = vr.next().asnumpy()
        if i in want:
            lin[i] = f
    del vr
    import av

    got = {}
    with av.open(path) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = frame.to_ndarray(format="rgb24")
            if i >= max(idx):
                break
    diff = [int(np.abs(got[i].astype(np.int16) - d[k]).max()) if i in got and got[i].shape == d[k].shape else None
            for k, i in enumerate(idx)]
    return {"indices": idx, "pixel_equal": all(x == 0 for x in diff), "n_equal": sum(x == 0 for x in diff),
            "max_abs_diff": max((x for x in diff if x is not None), default=None),
            "decord_seek_equals_linear": all(np.array_equal(lin[i], d[k]) for k, i in enumerate(idx))}


def scan_video(path: str, video_id: str, table_fps, compare_frames: bool = False) -> dict:
    import torch  # noqa: F401 - on Windows, loading decord before torch breaks torch's c10.dll

    p = Path(path)
    row: dict = {"video_id": video_id, "file": p.name, "exists": p.is_file(),
                 "authors_resolver_finds": (p.parent / f"{R.clean_text(video_id)}.mp4").is_file(),
                 "table_fps": table_fps}
    if not row["exists"]:
        return row
    try:
        import decord

        row["decord_frames"] = len(decord.VideoReader(path, num_threads=1))
    except Exception as e:  # noqa: BLE001
        row["decord_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        import av

        with av.open(path) as c:
            s = c.streams.video[0]
            row |= {"width": s.width, "height": s.height, "codec": s.codec_context.name,
                    "container_fps": float(s.average_rate) if s.average_rate else None}
            row["pyav_packets"] = sum(1 for pk in c.demux(s) if pk.size > 0)
    except Exception as e:  # noqa: BLE001
        row["pyav_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    n = row.get("decord_frames")
    if n is not None:
        row["plan"] = sampling_plan(n)
        k = row["plan"]["nframes"]
        if compare_frames and k and row["plan"]["case"] == "ok":  # short clips go through the torchvision fallback
            row["frames"] = _compare_frames(path, sample_indices(n, k))
    return row


def scan(df: pd.DataFrame, compare_frames: bool = False) -> dict:
    per_video = df.groupby("video_id").agg(video_path=("video_path", "first"), fps=("fps", "first"), n_questions=("qid", "size"))
    rows = []
    for vid, r in per_video.iterrows():
        row = scan_video(r.video_path, vid, int(r.fps), compare_frames) | {"n_questions": int(r.n_questions)}
        rows.append(row)
        print(f"[videoscan] {vid}: {row.get('decord_frames')} frames, {row.get('plan', {}).get('case')}", flush=True)
    t = pd.DataFrame(rows)

    def ids(mask) -> list[str]:
        return sorted(t.loc[mask, "video_id"].tolist())

    dec = t.get("decord_frames", pd.Series(np.nan, index=t.index))
    pyav = t.get("pyav_packets", pd.Series(np.nan, index=t.index))
    case = t.get("plan", pd.Series([{}] * len(t))).map(lambda p: p.get("case") if isinstance(p, dict) else None)
    cfps = pd.to_numeric(t.get("container_fps", pd.Series(np.nan, index=t.index)), errors="coerce")
    fr = t.get("frames", pd.Series([None] * len(t)))
    summ = {
        "n_videos": int(len(t)), "n_questions": int(t.n_questions.sum()),
        "missing_files": ids(~t.exists),
        "authors_resolver_misses": ids(t.exists & ~t.authors_resolver_finds),
        "decord_open_failures": ids(dec.isna() & t.exists),
        "decord_vs_pyav_count_mismatch": ids(dec.notna() & pyav.notna() & (dec != pyav)),
        "short_clips_authors_retry": ids(case == "authors_retry"),
        "short_clips_authors_retry_raises": ids(case == "authors_retry_raises_port_deviation"),
        "too_short_for_any_retry": ids(case == "too_short_for_any_retry"),
        "table_vs_container_fps_differ": ids(cfps.notna() & ((cfps - t.table_fps).abs() > 0.5)),
        "resolutions": t.apply(lambda r: f"{r.get('width')}x{r.get('height')}", axis=1).value_counts().to_dict(),
    }
    if "codec" in t:
        summ["codecs"] = t.codec.value_counts(dropna=False).to_dict()
    if compare_frames:
        is_cmp = fr.map(lambda f: isinstance(f, dict))
        summ["frames_compared_videos"] = int(is_cmp.sum())
        summ["decord_seek_differs_from_linear"] = ids(fr.map(lambda f: isinstance(f, dict) and not f["decord_seek_equals_linear"]))
        summ["frames_not_pixel_equal_to_pyav"] = ids(fr.map(lambda f: isinstance(f, dict) and not f["pixel_equal"]))
        summ["frames_max_abs_diff_to_pyav"] = max((f["max_abs_diff"] for f in fr[is_cmp] if f["max_abs_diff"] is not None),
                                                  default=None)
    questions_affected = {k: int(t.loc[t.video_id.isin(v), "n_questions"].sum()) for k, v in summ.items()
                          if isinstance(v, list) and v}
    return {"summary": summ | {"questions_affected": questions_affected},
            "videos": json.loads(t.to_json(orient="records"))}


def read_input_columns(path: Path) -> pd.DataFrame:
    """The id (first column, if it is not an input column) and INPUT_COLUMNS of a QuantiPhy CSV,
    as text. Nothing else is parsed: the answer column never reaches memory as values."""
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    id_col = header[0] if header[0] not in INPUT_COLUMNS else None
    use = ([id_col] if id_col is not None else []) + [c for c in INPUT_COLUMNS if c in header]
    df = pd.read_csv(path, usecols=use, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = df.rename(columns={id_col: "id"}) if id_col is not None else df.assign(id=[str(i) for i in range(len(df))])
    for c in INPUT_COLUMNS:
        if c in df:
            df[c] = df[c].map(R.clean_text)  # the authors' loader cleans every cell the same way
    return df[["id"] + [c for c in INPUT_COLUMNS if c in df]]


def diff_input_columns(local: pd.DataFrame, github: pd.DataFrame) -> dict:
    """Row match by id when the ids overlap, else by position; per column: differing cells and examples."""
    by_id = len(set(local.id) & set(github.id)) > 0
    if by_id:
        a, b = local.set_index("id"), github.set_index("id")
        common = a.index.intersection(b.index)
        only_local, only_github = sorted(a.index.difference(b.index)), sorted(b.index.difference(a.index))
        a, b = a.loc[common], b.loc[common]
    else:
        n = min(len(local), len(github))
        a, b = local.iloc[:n].set_index("id"), github.iloc[:n].set_index(local.iloc[:n].id)
        only_local, only_github = list(local.id.iloc[n:]), list(github.id.iloc[n:])
    cols = {}
    for c in INPUT_COLUMNS:
        if c not in a or c not in b:
            cols[c] = {"missing_in": "local" if c not in a else "github"}
            continue
        x, y = a[c], b[c]
        if c == "fps":
            differ = pd.to_numeric(x, errors="coerce").ne(pd.to_numeric(y, errors="coerce")) | (x.eq("") != y.eq(""))
        else:
            differ = x.ne(y)
        cols[c] = {"n_differ": int(differ.sum()),
                   "examples": [{"id": i, "local": x[i], "github": y[i]} for i in x.index[differ][:5]]}
    return {"matched_by": "id" if by_id else "row order", "n_local": int(len(local)), "n_github": int(len(github)),
            "n_compared": int(len(a)), "only_local": only_local[:20], "only_github": only_github[:20], "columns": cols}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--compare-frames", action="store_true", help="decord vs PyAV pixels at the sampled indices")
    ap.add_argument("--github-csv", default=None, help="val only: GitHub quantiphy_validation.csv (diff input columns)")
    args = ap.parse_args(argv)
    if args.github_csv is not None:
        if args.split != "val":
            ap.error("--github-csv is for --split val")
        out = diff_input_columns(read_input_columns(VAL_CSV), read_input_columns(Path(args.github_csv)))
        out |= {"local": VAL_CSV.relative_to(ROOT).as_posix(), "github": str(args.github_csv), "columns_read": INPUT_COLUMNS}
        p = ROOT / "results" / "caw_val_github_csv_diff.json"
    else:
        from methods.caw.run import questions  # drops validation answers

        out = scan(questions(args.split), args.compare_frames)
        p = ROOT / "results" / f"caw_{args.split}_video_scan.json"
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out.get("summary", out), indent=1, default=str))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
