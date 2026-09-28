# QuantiPhy 2026: open-weight entry (Track B)

## QuantiPhy

The [QuantiPhy Challenge 2026](https://quantiphy.stanford.edu/competition/) (a NeurIPS 2026
competition) asks for a physical quantity (a size, distance, speed or acceleration) from a
short video. Each question comes with one known quantity in the scene (the "prior", for
example `length of boat = 3.62m`) and, for 3D scenes, depth information. Answers are scored
with Mean Relative Accuracy (MRA), averaged over four categories: S2, D2, S3, D3
(a static or dynamic prior, in a 2D or 3D scene).

## What this repository is

This is the code of our **Track B (Open-Weight)** entry, `hybrid_v4`. The pipeline uses
only open-weight models that run locally; no closed-API model is called anywhere.

Official hidden-test score of the submitted file `hybrid_v4_test.csv`, as shown on the
competition site (uploaded with "Track B · Open weight" selected on 2026-09-29):

| MRA | S2 | D2 | S3 | D3 |
|---|---|---|---|---|
| **0.552** | 0.532 | 0.593 | 0.576 | 0.508 |

## Method

1. **Geometry measurement (`methods/geometry`, "geometry_v1").** The question and the
   prior are parsed into the objects to find and the quantity to measure. OWLv2 detects
   those objects by text query on up to 96 uniformly sampled frames per video; the
   detections are linked into tracks, and sizes, displacements, speeds and accelerations
   are measured in pixels. The prior sets the metric scale (2D: metres per pixel; 3D: a
   focal length calibrated with the given depths). When a question cannot be measured, a
   label-free fallback is used (the prior value if it has the same dimension, otherwise a
   median of the test-set priors).
2. **VLM answer (`methods/vlm_baseline`).** Qwen3-VL-8B-Instruct sees 32 uniformly sampled
   frames with timestamps, the prior, the depth information and the question (prompt wording
   from the official starter kit) and replies with a number and a unit. The first number is
   used, converted when the reply names a different unit of the same dimension (for example
   cm instead of m); with no usable number, the geometry fallback value is used.
3. **Box verification (`methods/hybrid_v2/verify.py`).** For every track the geometry answer
   depends on, Qwen3-VL-8B is shown 3 frames (the full frame with the box in red plus a zoomed
   crop) and asked whether the box contains the named object; `p_yes` is read from the
   next-token probabilities.
4. **Arbitration, hybrid_v2 (`methods/hybrid_v2/hybrid.py`).** Use the geometry answer when
   geometry measured the question and every track it used passed the box check (mean
   `p_yes` >= 0.5). Otherwise use the VLM answer; if the box check failed and the VLM also
   gave no usable number, keep the geometry answer. Thresholds are in `methods/hybrid_v2/config.json`.
5. **Larger VLM, hybrid_v4 (`methods/vlm_large`).** Every question that hybrid_v2 answered
   with the 8B VLM is asked again with the same prompt, frames and decoding, using
   Qwen3-VL-32B-Instruct quantized to 4-bit on load (bitsandbytes NF4, double quantization,
   bf16 compute, all modules quantized so it fits in 24 GB). Its answer replaces the 8B answer
   unless it gave no usable number. All other answers stay as in hybrid_v2. Nothing is tuned.

On the test set, hybrid_v2 answered 2,483 questions with geometry, 9 with geometry after a
failed box check (the VLM gave no usable number), and 797 with the VLM (500 where geometry fell back,
297 where a box was rejected). In hybrid_v4, the 32B model answered 708 of those 797; for
the other 89 it gave no usable number and the hybrid_v2 answer was kept.

## Models (all pinned to a Hugging Face revision in the code)

| Model | Revision | Used in | License |
|---|---|---|---|
| [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) | `cfd3195ba4ea9592eec887ded089f4c08eff231d` | geometry detection, fp16 | Apache-2.0 |
| [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | VLM answers and box verification, bf16, greedy | Apache-2.0 |
| [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | hybrid_v4 VLM answers, 4-bit NF4, greedy | Apache-2.0 |

Licenses are as stated in each model card's metadata at the pinned revision. No weights are
included here; `transformers` downloads them on first use. The environment is locked in
`uv.lock` (Python 3.11, torch 2.11.0+cu128, transformers 4.57.6, bitsandbytes 0.50.2,
accelerate 1.15.0).

## Hardware and run times

All runs used one NVIDIA RTX 4090 (24 GB) on Windows 11. Each GPU step checks free GPU memory
before it starts: 8 GiB for OWLv2 detection, 19 GiB for the Qwen3-VL-8B steps and 21 GiB for
Qwen3-VL-32B, so the 32B step needs a 24 GB GPU with little else running. Recorded for the
test set:

- Qwen3-VL-8B answers: 3,289 questions, 58.1 min of model inference in total, peak reserved
  VRAM 19.0 GiB.
- Qwen3-VL-32B answers: 797 questions, 47.0 min of model inference in total, peak reserved
  VRAM 21.9 GiB.

These figures exclude model loading and video decoding. Detection and box-verification times
were not stored with the results. The two combine steps (`hybrid_v2.hybrid` and
`vlm_large.hybrid`) run on CPU.

## Reproduce

Install [uv](https://docs.astral.sh/uv/) and git, then run everything from the repository
root. All paths are relative to the repository root.

### 1. Setup

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

This runs the three steps below. On Linux (not tested; `uv.lock` includes the Linux wheels),
run them directly:

```bash
uv sync                                    # Python 3.11 env in .venv (torch from the CUDA 12.8 index)
uv run python scripts/download_data.py     # datasets + submission template -> data/
git clone https://github.com/Paulineli/QuantiPhy external/QuantiPhy
git -C external/QuantiPhy checkout 4f9323c9ca9479fc673749ae7d2a82729fef6e85
```

`scripts/download_data.py` downloads the official data at pinned revisions:
[PaulineLi/QuantiPhy](https://huggingface.co/datasets/PaulineLi/QuantiPhy) (test set) at
`a640cf78a9ac07b17a270a63372f6f65ac95b8a9`,
[PaulineLi/QuantiPhy-validation](https://huggingface.co/datasets/PaulineLi/QuantiPhy-validation)
at `74aec82473912ccab6a589645a39fcd0e78ee23c`, and the submission template from the
competition site. The starter kit (official `evaluator.py`) is only needed for the tests and
for validation scoring.

### 2. Pipeline (test split)

Run in this order. GPU steps cache per-video or per-question results under `runs/` and
resume where they stopped.

```bash
uv run python -m methods.geometry.run --split test      # GPU (OWLv2) + CPU solve -> submissions/geometry_v1_test.csv
uv run python -m methods.vlm_baseline.run --split test  # GPU (Qwen3-VL-8B)        -> submissions/vlm_baseline_test.csv
uv run python -m methods.hybrid_v2.verify --split test  # GPU (Qwen3-VL-8B box check)
uv run python -m methods.hybrid_v2.hybrid --split test  # CPU                      -> submissions/hybrid_v2_test.csv
uv run python -m methods.vlm_large.run --split test     # GPU (Qwen3-VL-32B, 4-bit; needs the previous step)
uv run python -m methods.vlm_large.hybrid --split test  # CPU                      -> submissions/hybrid_v4_test.csv
```

`submissions/hybrid_v4_test.csv` is the Track B file: the official template with
`parsed_value` filled in. Each step also writes a summary to `results/<name>_test_summary.json`;
`results/hybrid_v4_test_summary.json` should show the source counts above
(`v2` 2492, `vlm32` 708, `v2:vlm32_fell_back` 89). GPU inference is not guaranteed to be
bit-exact on other GPUs or drivers; the CPU steps are deterministic given the cached GPU
outputs in `runs/`.

### 3. Validation (optional)

The same six commands with `--split val` produce `results/predictions/val_*.csv`.
Score a file with our scorer and the official evaluator side by side:

```bash
uv run python -c "import json; from quantiphy.parity import compare_with_official as c; print(json.dumps(c('results/predictions/val_hybrid_v4.csv'), indent=1))"
```

### 4. Tests

```bash
uv run pytest -q
```

The tests need `data/` and `external/QuantiPhy` from the setup step; no GPU is used.

`runs/`, `submissions/` and `results/` contain test-set predictions (many validation questions
also appear in the test set), so they are git-ignored. Please keep them private while the
competition is running.

## Notes

- **Validation did not show hybrid_v4 to be better than hybrid_v2.** The v4 rule was fixed
  before the validation set was scored. On the 159 validation questions, hybrid_v4 scored
  0.5006 MRA and hybrid_v2 0.5044: a difference of -0.0038, with a 95% interval of
  [-0.0237, +0.0138] from a paired, video-level bootstrap (10,000 resamples).
- **hybrid_v4 was chosen after seeing hidden-test scores.** It is our Track B entry because it
  has the highest official hidden-test score among our open-weight uploads:

  | Version | MRA | S2 | D2 | S3 | D3 |
  |---|---|---|---|---|---|
  | geometry_v1 | 0.516 | 0.507 | 0.522 | 0.563 | 0.472 |
  | hybrid_v1 (geometry, 8B answer where geometry fell back)* | 0.531 | 0.516 | 0.559 | 0.565 | 0.483 |
  | hybrid_v2 | 0.542 | 0.524 | 0.584 | 0.569 | 0.490 |
  | **hybrid_v4** | **0.552** | 0.532 | 0.593 | 0.576 | 0.508 |

  \* hybrid_v1 is hybrid_v2 without the box check; its combine script is not part of this repository.
- **No closed-API model is used anywhere in this pipeline.** The starter kit ships GPT-5.1
  outputs for the validation set (`external/QuantiPhy/model_outputs/gpt-5.1.csv`). They were
  used only as a reference score during development, and the tests use them as input to check
  our MRA scorer against the official evaluator and to exercise the bootstrap code. No pipeline
  step reads them.
- The fallback constants (medians of the test-set priors: 0.438 m, 1.4 m/s, 9.7 m/s²) are
  computed from the prior text of the test questions only.

## Data and third-party material

- The QuantiPhy datasets are licensed CC-BY-4.0 (per the dataset cards at the pinned
  revisions). They are **not** redistributed here; `scripts/download_data.py` fetches them from
  Hugging Face and the submission template from the competition site.
- The Qwen3-VL-8B prompt reuses wording from the official starter kit (MIT License, Copyright
  (c) 2025 Puyin Li); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Model weights are not included; see the model table above.

## License

The code in this repository is released under the MIT License; see [LICENSE](LICENSE).
