# quantiphy-geo-vlm: QuantiPhy 2026 open-weight entry (Track B)

## QuantiPhy

The [QuantiPhy Challenge 2026](https://quantiphy.stanford.edu/competition/) (a NeurIPS 2026
competition) asks for a physical quantity (a size, distance, speed or acceleration) from a
short video. Each question comes with one known quantity in the scene (the "prior", for
example `length of boat = 3.62m`) and, for 3D scenes, depth information. Answers are scored
with Mean Relative Accuracy (MRA): each answer scores the fraction of ten relative-error
tolerances (0.9, 0.8, ..., 0.1 and 0.05) that it meets. Scores are averaged within each of
four categories, S2, D2, S3 and D3 (a static or dynamic prior, in a 2D or 3D scene), and then
over the four categories.

## What this repository is

This is the code of our **Track B (Open-Weight)** uploads. Every model used here has open weights
and runs locally; no closed-API model is called anywhere.

Official test-set scores (answers withheld by the organizers), as shown on the competition site.
After the rules clarification of 2026-10-15 the organizers rescored all uploads on the 3,010 test
questions that remain after their exclusions (originally 3,289); the main columns below are these
rescored values, and the last column is the score first shown on the original 3,289 questions.
hybrid_v4, caw and caw_geospeed were uploaded with "Track B · Open weight" selected on 2026-09-29
(hybrid_v4 had already been uploaded on 2026-09-24, with the same score). `caw_geospeed_iq_test.csv`
was uploaded with Track B selected on 2026-10-25; it is our last Track B upload and our **final
Track B entry**.

| Version | Answers from | MRA (3,010) | S2 | D2 | S3 | D3 | MRA (3,289) |
|---|---|---|---|---|---|---|---|
| hybrid_v4 | our geometry measurement + Qwen3-VL-8B/32B, arbitrated | 0.551 | 0.522 | 0.594 | 0.579 | 0.511 | 0.552 |
| caw | Code-as-World-VL-9B (a third-party open fine-tune) on every question; hybrid_v4 for 2 unusable replies | 0.542 | 0.473 | 0.582 | 0.550 | 0.565 | 0.544 |
| caw_geospeed (earlier entry) | caw, with our geometry answers on 708 speed questions; not per question | 0.579 | 0.539 | 0.593 | 0.614 | 0.570 | 0.577 |
| **caw_geospeed_iq** (final) | per-question version: Code-as-World-VL-9B, with per-question geometry answers on 718 speed questions | **0.578** | 0.537 | 0.591 | 0.616 | 0.566 | - |

Most answers of the final entry (2,571 of 3,289) come from
[Code-as-World-VL-9B](https://huggingface.co/MirroS-Lab/Code-as-World-VL-9B), a model fine-tuned
and released by its authors (MirroS-Lab); the other 718 are this repository's geometry
measurements, made per question. caw_geospeed_iq answers every question from that question's own
inputs only, as the 2026-10-15 rules clarification asks; caw_geospeed did not (its geometry answers
used other questions of the same video), so it is not our entry, although it scores 0.001 higher on
the rescored questions. caw_geospeed_iq was fixed as the final entry before its test score was
known; see [Per-question version](#per-question-version-caw_geospeed_iq-and-the-fallback-caw_iq).
Before that, caw_geospeed had been kept as the entry after seeing the test scores of hybrid_v4, caw
and caw_geospeed; see [Notes](#notes).

**Code state.** The hybrid_v4 code is the version that produced the submitted file. For
publication, comments and docstrings were edited and a few `mkdir` calls were added so that
missing output folders are created; no prediction changes. The OWLv2 revision was pinned in
`methods/geometry/detect.py` after the test-set detection run, to the same snapshot that run used.
From the cached GPU outputs, the CPU combine steps reproduce the submitted file byte for byte (on
Windows; see [Pipeline](#2-hybrid_v4-pipeline-test-split)). `methods/caw`, `methods/common`,
`envs/caw`, `tests/test_caw.py` and `scripts/eval_caw_val.py` are the code that produced
`caw_test.csv` and `caw_geospeed_test.csv`. For publication, docstrings that referred to internal
documents were edited, Apache-2.0 notices were added to the files adapted from Code-as-World, one
test was rewritten as an import check, and `envs/caw/uv.lock` was re-resolved against this
repository's root `pyproject.toml` (all 72 locked packages keep the same versions and file
hashes); apart from the wording of one low-memory warning, code behaviour is unchanged.
`methods/caw/run.py` stamps every record with a hash of the files that decide the answers
(`RUN_CODE_PATHS`), so records made from this repository carry a different `code_sha256` than
ours. Of those 15 files, 11 are identical to the ones that made our records; `methods/caw/run.py`,
`methods/caw/recipe.py` and `methods/common/parse_sci.py` differ only in the edits above
(comments, docstrings, one warning message), and `envs/caw/uv.lock` as described above.
`methods/geometry_iq`, `tests/test_geometry_iq.py`, `scripts/check_geometry_iq_detect.py`,
`scripts/geometry_iq_wait_run.sh` and `scripts/eval_caw_iq_val.py` are the code that produced
`caw_geospeed_iq_test.csv` (added on 2026-10-25). For publication, the references to our internal
plan (in docstrings, in the `--prereg` examples and as the plan path checked and recorded by
`scripts/eval_caw_iq_val.py`) now name this README, and the runner script changes to the
repository folder with a relative path; the code logic is unchanged. Because `methods/geometry_iq/run.py` is one of the hashed files, its `code_sha256`
also differs from ours.

## Method

### hybrid_v4

1. **Geometry measurement (`methods/geometry`, "geometry_v1").** The question and the
   prior are parsed into the objects to find and the quantity to measure. OWLv2 detects
   those objects by text query on up to 96 uniformly sampled frames per video; the
   detections are linked into tracks, and sizes, displacements, speeds and accelerations
   are measured in pixels. The prior sets the metric scale: in 2D, metres per pixel; in 3D,
   a focal length calibrated so that the prior matches the given depths, or an assumed
   ~64° horizontal field of view when that calibration is not possible (392 test questions).
   A 3D target without depth information uses the prior's pixel scale (size priors) or the
   median annotated depth of the scene. When a question cannot be measured, a label-free
   fallback is used (the prior value if it has the same dimension, otherwise a median of the
   test-set priors).
2. **VLM answer (`methods/vlm_baseline`).** Qwen3-VL-8B-Instruct sees 32 uniformly sampled
   frames with timestamps, the prior, the depth information and the question, and replies
   with a number and a unit. The system prompt, the context wording and the closing
   instruction come from the official starter kit, plus one added sentence giving the fps,
   the clip length and the number of frames shown. The first number is used, converted when
   the reply names a different unit of the same dimension (for example cm instead of m); with
   no usable number (no finite number, or 0), the geometry fallback value is used.
3. **Box verification (`methods/hybrid_v2/verify.py`).** For every track the geometry answer
   depends on, at up to 3 frames of the track (20%, 50% and 80% along it), Qwen3-VL-8B is
   shown the frame with the box outlined in red plus a zoomed crop of the box, and asked
   whether the box contains the named object. `p_yes` = P(yes) / (P(yes) + P(no)) is read
   from the next-token probabilities (one forward pass per frame) and averaged over the frames.
4. **Arbitration, hybrid_v2 (`methods/hybrid_v2/hybrid.py`).** Use the geometry answer when
   geometry measured the question and every track it used passed the box check (mean
   `p_yes` >= 0.5; a track with no checkable frame counts as passed). Otherwise use the VLM
   answer; if the box check failed and the VLM also gave no usable number, keep the geometry
   answer. Thresholds are in `methods/hybrid_v2/config.json`.
5. **Larger VLM, hybrid_v4 (`methods/vlm_large`).** Every question that hybrid_v2 answered
   with the 8B VLM is asked again with the same prompt, frames and decoding, using
   Qwen3-VL-32B-Instruct quantized to 4-bit on load (bitsandbytes NF4, double quantization,
   bf16 compute, all modules quantized so it fits in 24 GB). Its answer replaces the 8B answer
   unless it gave no usable number. All other answers stay as in hybrid_v2. Nothing is tuned.

On the test set, hybrid_v2 answered 2,483 questions with geometry, 9 with geometry after a
failed box check (the VLM gave no usable number), and routed 797 to the VLM: 500 where
geometry fell back (for 58 of these the 8B reply was not usable either, so the label-free
fallback was used) and 297 where a box was rejected. In hybrid_v4, the 32B model answered 708
of those 797; for the other 89 it replied 0, so the hybrid_v2 answer was kept.

### Code-as-World variants: caw and caw_geospeed

1. **Code-as-World-VL-9B answers (`methods/caw/run.py`, `methods/caw/recipe.py`).**
   Code-as-World-VL-9B (fine-tuned by its authors from Qwen3.5-9B) answers every question in
   bf16 with the authors' QuantiPhy recipe from
   [MirroS-Lab/Code-as-World](https://github.com/MirroS-Lab/Code-as-World) (commit `1353bf07`):
   their system prompt, "Given that <prior>. [depth information] <question>" in their format
   template, their chat template with thinking off, 16 frames read with qwen-vl-utils and decord
   (at most 262,144 pixels per frame), timestamps from the dataset's fps, and at most 512 new
   tokens. The authors run the model with vLLM on Linux; vLLM does not run on Windows, so the
   recipe is ported to Hugging Face transformers `generate()`: argmax decoding (their temperature
   0.01 / top_p 0.001 keeps only the top token), with the prompt and video tokens laid out as vLLM
   lays them out. Each reply is read with `parse_answer_sci` (`methods/common/parse_sci.py`: the
   VLM parser of hybrid_v4 that also reads numbers written as `2.27×10³`) and, for comparison,
   with the authors' parser (first number, no unit handling). The ×10ⁿ reading was written
   earlier, during our Track A work, after reading reply texts (never answers) of another model;
   here the two parsers read the same number from every validation and test reply, so it changed
   no answer.
2. **caw (`methods/caw/combine.py`, CPU).** Every question takes the absolute value of the
   Code-as-World answer when the reply gives a finite, non-zero number, otherwise its hybrid_v4
   answer. On the test set: 3,287 Code-as-World answers and 2 hybrid_v4 answers (two S3 replies
   of 0).
3. **caw_geospeed.** As caw, except that speed questions whose hybrid_v2 source is exactly
   `geometry` (geometry measured the question and every box passed the check) take that geometry
   answer, which is also hybrid_v4's answer there. On the test set: 708 geometry answers (S2 169,
   D2 172, S3 204, D3 163; not the same questions as the 708 that the 32B model answered in
   hybrid_v4, which hybrid_v2 had routed to the VLM) and 2,581 Code-as-World answers.

No threshold or setting was tuned, and both rules were fixed before the validation run; the speed
rule itself, however, came from an earlier validation analysis (see [Notes](#notes)).
Differences from the authors' run are listed in `PORT_DIFFERENCES` in `methods/caw/run.py`
(also written into every run summary). The main ones: another engine on another operating
system, whose bf16 numerics can flip an argmax token and so change a reply; argmax instead of
their near-greedy sampling (the two differ only on exact probability ties); and one test video of
15 frames (`internet_0038`) on which the authors' short-clip retry raises again and their run
would stop, so a second retry uses 14 frames.

### Per-question version: caw_geospeed_iq (and the fallback caw_iq)

On 2026-10-15 the organizers asked every team to "answer each question independently, without
using other questions or their associated answers or priors to infer its answer". The
Code-as-World answers of caw_geospeed are per question, but its 708 geometry speed answers were
not: geometry_v1 runs OWLv2 once per video with the object names of every question of that video
(so a box can be suppressed by another question's name), picks free-fall candidates from all
questions of the video, uses the median prior of the whole test set as a plausibility reference,
and hybrid_v2 checks each (video, track) once, with the wording of the first question that used
it. `methods/geometry_iq` rebuilds that path per question; `methods/geometry` and
`methods/hybrid_v2` are imported unchanged:

1. **Detection per question (GPU, OWLv2, same weights, frames and settings).** One detector call
   per question, whose text queries are only the object names in that question and in its own
   prior; the video is decoded per question as well. Cross-query box suppression therefore only
   involves the question's own names, and free-fall candidates are only its own objects.
2. **Fixed reference constants** for the plausibility check and the fallback: the median |prior|
   per dimension over the *validation* priors (0.94 m, 1.25 m/s, 4.0 m/s²), for both splits.
3. **Box check per question (GPU, Qwen3-VL-8B, same prompt, frames and 0.5 threshold).** Every
   track a question used is checked with that question's own text.
4. **caw_geospeed_iq (`methods/geometry_iq/combine.py`, CPU).** A question whose own text is a
   speed question and whose geometry measured with every box accepted takes the geometry answer;
   otherwise the Code-as-World answer; if that reply is unusable, the question's own
   same-dimension prior value, else the fixed constant above (no hybrid_v4 fallback, since
   hybrid_v4 contains the old geometry path). **caw_iq** is the same without geometry (the
   fallback file, no GPU needed).

On the test set caw_geospeed_iq has 2,571 Code-as-World and 718 geometry answers and differs from
caw_geospeed on 108 questions: of the 708 old geometry answers, 694 are still geometry (70 with a
different value) and 14 are now Code-as-World answers; 24 speed questions that were Code-as-World
answers are now geometry answers. `tests/test_geometry_iq.py` checks on real validation and test
questions that removing or altering the other questions of the same video (their text, prior,
detections, Code-as-World answers, geometry sources) does not change a question's answer, and
`scripts/check_geometry_iq_detect.py` re-ran detection on a GPU for 12 test questions right after
a call with the other questions' names: boxes and scores were bit-identical in all 12. The plan
(fixed before the validation run) uploads caw_geospeed_iq as the final Track B entry whatever its
score. Validation (159 questions, scored once): caw_geospeed_iq 58.03 MRA vs caw_geospeed 58.42
(video-level 95% CI of the difference [-1.18, +0.12]); caw_iq 55.18, equal to caw. Official test
score (uploaded 2026-10-25, scored on the 3,010 rescored questions): MRA 0.578 (S2 0.537, D2 0.591,
S3 0.616, D3 0.566), against 0.579 for caw_geospeed on the same questions. The fixed reference
constants come from the public validation priors (prior text only, no answers) and are used only
in the 1000x plausibility check and in the fallback; on the test set no caw_geospeed_iq answer is
a fallback value.

## Models (all pinned to a Hugging Face revision in the code)

| Model | Revision | Used in | License |
|---|---|---|---|
| [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) | `cfd3195ba4ea9592eec887ded089f4c08eff231d` | geometry detection, fp16 | Apache-2.0 |
| [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | VLM answers (bf16, greedy) and box verification (bf16, one forward pass per frame) | Apache-2.0 |
| [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | hybrid_v4 VLM answers, 4-bit NF4, greedy | Apache-2.0 |
| [MirroS-Lab/Code-as-World-VL-9B](https://huggingface.co/MirroS-Lab/Code-as-World-VL-9B) | `46b111c53f5680eb12c63a7391e5c690d1e14ab1` | caw / caw_geospeed / caw_geospeed_iq answers (bf16, argmax) | Apache-2.0 |

Licenses are as stated in each model card's metadata at the pinned revision. No weights are
included here. The main environment is locked in `uv.lock` (Python 3.11, torch 2.11.0+cu128,
transformers 4.57.6, bitsandbytes 0.50.2, accelerate 1.15.0). Its transformers cannot load the
Code-as-World checkpoint, so the Code-as-World variants have their own environment, locked in
`envs/caw/uv.lock` (Python 3.11, torch 2.10.0+cu128, torchvision 0.25.0+cu128, transformers
5.11.0, qwen-vl-utils 0.0.14, decord 0.6.0, accelerate 1.15.0): the authors' inference stack
without vLLM, with the torch version vLLM 0.19.1 pins.

## Hardware and run times

All runs used one NVIDIA RTX 4090 (24 GB) on Windows 11. Each GPU step checks free GPU memory
before it starts: 8 GiB for OWLv2 detection, 19 GiB for the Qwen3-VL-8B steps, 21 GiB for
Qwen3-VL-32B and 20.5 GiB for Code-as-World-VL-9B, so the last two need a 24 GB GPU with little
else running. Recorded for the test set:

- Qwen3-VL-8B answers: 3,289 questions, 58.1 min of model inference in total, peak reserved
  VRAM 19.0 GiB.
- Qwen3-VL-32B answers: 797 questions, 47.0 min of model inference in total, peak reserved
  VRAM 21.9 GiB.
- Code-as-World-VL-9B answers: 3,289 questions, 70.4 min of model inference in total (median
  0.86 s per question) plus 4.3 min of video preparation, peak reserved VRAM 19.1 GiB. The run
  took about 1 h 21 min wall-clock, model loading included, without interruption.

The first two figures exclude model loading and video decoding. Detection and box-verification
times were not stored with the results. The combine steps (`hybrid_v2.hybrid`,
`vlm_large.hybrid` and `caw.combine`) run on CPU.

## Reproduce

Install [uv](https://docs.astral.sh/uv/) (tested with uv 0.11.18) and git, then clone this
repository and run everything from its root. All paths are relative to the repository root.

```bash
git clone https://github.com/kuotunyu/quantiphy-geo-vlm
cd quantiphy-geo-vlm
```

The GPU steps need an NVIDIA GPU with 24 GB and a driver that supports CUDA 12.8, on Windows
or Linux (only Windows was tested). macOS is not supported: the lock files have the CUDA builds
of torch for Windows and Linux only. Weights are downloaded from Hugging Face; no Hugging Face
token is needed, as none of the models or datasets is gated.

### 1. Setup

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

This runs the three steps below. On Linux, run them directly:

```bash
uv sync --locked                           # Python 3.11 env in .venv, exactly as in uv.lock (torch from the CUDA 12.8 index)
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

### 2. hybrid_v4 pipeline (test split)

Run in this order. GPU steps cache per-video or per-question results under `runs/` and
resume where they stopped. The 32B step downloads the full bf16 weights and quantizes them on
load.

```bash
uv run python -m methods.geometry.run --split test      # GPU (OWLv2) + CPU solve -> submissions/geometry_v1_test.csv
uv run python -m methods.vlm_baseline.run --split test  # GPU (Qwen3-VL-8B)        -> submissions/vlm_baseline_test.csv
uv run python -m methods.hybrid_v2.verify --split test  # GPU (Qwen3-VL-8B box check)
uv run python -m methods.hybrid_v2.hybrid --split test  # CPU                      -> submissions/hybrid_v2_test.csv
uv run python -m methods.vlm_large.run --split test     # GPU (Qwen3-VL-32B, 4-bit; needs the previous step)
uv run python -m methods.vlm_large.hybrid --split test  # CPU                      -> submissions/hybrid_v4_test.csv
```

`submissions/hybrid_v4_test.csv` is the official template with `parsed_value` filled in. Every
step except `hybrid_v2.verify` (which only writes its cache under `runs/`) also writes a summary
to `results/<name>_test_summary.json`; `results/hybrid_v4_test_summary.json` should show the
source counts above (`v2` 2492, `vlm32` 708, `v2:vlm32_fell_back` 89).

### 3. Code-as-World variants (test split)

The combine step needs the hybrid_v4 outputs of section 2 (`submissions/hybrid_v4_test.csv` and
`runs/hybrid_v2/test/items.csv`). Run section 2 first; if you run `methods.caw.run` on a fresh
clone before it, create the `results/` folder first (`mkdir results`), or the run stops with an
error when it writes its summary at the end (the per-question records are kept, so after creating
the folder, running the same command again only collects them and writes the summary).

```bash
uv sync --project envs/caw --locked       # separate env in envs/caw/.venv, exactly as in envs/caw/uv.lock
# the checkpoint (18.8 GB); methods/caw/run.py loads it from the local Hugging Face cache only
uv run --project envs/caw python -c "from huggingface_hub import snapshot_download; snapshot_download('MirroS-Lab/Code-as-World-VL-9B', revision='46b111c53f5680eb12c63a7391e5c690d1e14ab1')"
uv run --project envs/caw python -m methods.caw.run --split test   # GPU -> runs/caw/test/items.csv, results/caw_test_summary.json
uv run python -m methods.caw.combine --split test                  # CPU, main env -> submissions/caw_test.csv, submissions/caw_geospeed_test.csv
```

`methods.caw.run` caches one record per question under `runs/caw/test/items/`; if it is
interrupted, run the same command again to answer only the pending questions
(`--max-videos N` stops after N videos, exit code 3 while videos are pending). It refuses to
start when the files that decide the answers (`RUN_CODE_PATHS` in `methods/caw/run.py`) have
uncommitted changes or when cached records were made by other code, so run it from a clean
checkout. `--limit-videos 2 --recheck 1` runs a short probe instead. `methods.caw.combine` refuses
to run if a record carries an error (a video the run gave up on) unless that video is named with
`--accept-failed-videos`; in our run no record had an error. `results/caw_combine_test_summary.json`
should show the source counts above (caw: `caw` 3287, `hybrid_v4:caw_unusable` 2; caw_geospeed:
`caw` 2581, `geometry:speed` 708).

GPU inference is not guaranteed to be bit-exact on other GPUs or drivers; the CPU steps are
deterministic given the cached GPU outputs in `runs/`. pandas writes the CSV files with the
operating system's line ending, and the template (like the submitted files) has CRLF row
endings, so the files match byte for byte only on Windows; on other systems compare the
`parsed_value` column instead.

### 3b. Per-question version (test split)

Needs `runs/caw/test/items.csv` from section 3 (not the hybrid outputs). `--prereg` names a
committed file recorded as the pre-registration (ours was an internal plan that is not included;
`README.md` works). The script waits until no other python process uses the GPU, then runs one
process per stage (OWLv2 detection, CPU solve, Qwen3-VL-8B box check, CPU decision) and the
combine step:

```bash
SPLIT=test bash scripts/geometry_iq_wait_run.sh   # -> runs/geometry_iq/test/, submissions/caw_geospeed_iq_test.csv, submissions/caw_iq_test.csv
# or stage by stage:
uv run python -m methods.geometry_iq.run --split test --stage detect --prereg README.md   # also: solve, verify, decide
uv run python -m methods.geometry_iq.combine --split test
uv run python -m methods.geometry_iq.combine --split test --insurance-only               # caw_iq only, CPU
```

`results/caw_iq_combine_test_summary.json` should show caw_geospeed_iq sources `caw` 2571 and
`geometry_iq:speed` 718. Our test run took 28.2 min for the last 587 of 894 detections (the run
was resumed twice; finished questions are not rerun) and 16.5 min for the box check of 1,071
tracks, model loading excluded. On validation, `scripts/eval_caw_iq_val.py` is the one-time
scoring (it reads `results/caw_val.json` from `scripts/eval_caw_val.py`).

### 4. Validation (optional)

The commands of sections 2 and 3 with `--split val` produce `results/predictions/val_*.csv`
(including `val_caw.csv`, `val_caw_geospeed.csv` and `val_caw_authors.csv`, the last one being
the authors' parser with no fallback). On validation, `methods.caw.run` also needs
`--prereg <file>`, a file committed in the repository that is recorded as the pre-registration
in every record (ours was an internal plan, `docs/PHASE11_PLAN.md`, which is not included).
Score a file with our scorer and the official evaluator side by side:

```bash
uv run python -c "import json; from quantiphy.parity import compare_with_official as c; print(json.dumps(c('results/predictions/val_caw_geospeed.csv'), indent=1))"
```

`scripts/eval_caw_val.py` is the one-time scoring we ran for the Code-as-World variants
(gate, bootstraps, comparison with the paper). It refuses to run if `results/caw_val.json`
already exists or if the scoring code has uncommitted changes, and it also reads
`results/hybrid_v4_val.json` (hybrid_v4's validation macro MRA under
`systems.hybrid_v4.macro_mra`), which is written by a scoring script that is not part of this
repository. To create it from `results/predictions/val_hybrid_v4.csv`:

```bash
uv run python -c "import json; from quantiphy.data import load_validation, read_predictions; from quantiphy.mra import score; m = score(load_validation(), read_predictions('results/predictions/val_hybrid_v4.csv')).macro; open('results/hybrid_v4_val.json', 'w').write(json.dumps({'systems': {'hybrid_v4': {'macro_mra': m}}}))"
```

The script also refuses uncommitted changes to `results/predictions/val_hybrid_v4.csv`; in this
repository `results/` is git-ignored, so that part of the check has no effect.

`methods.caw.videoscan` (optional, caw env) checks the video inputs without reading any answer:
whether each file exists, its frame count and fps, and what the 16-frame sampling does with short
clips. It writes `results/caw_<split>_video_scan.json` and does not affect any answer.

### 5. Tests

```bash
uv run pytest -q                                                  # main env
uv run --project envs/caw python -m pytest tests/test_caw.py -q   # Code-as-World port in its own env
```

The tests need `data/` and `external/QuantiPhy` from the setup step; no GPU is used. In
`tests/test_caw.py`, one test needs the caw env and the checkpoint's tokenizer files in the
Hugging Face cache, and the tests that compare the port with the authors' code need their
repository (otherwise they are skipped):

```bash
git clone https://github.com/MirroS-Lab/Code-as-World external/Code-as-World
git -C external/Code-as-World checkout 1353bf07d24e5463caff92ce23ffd34d03984831
```

`runs/`, `submissions/` and `results/` contain test-set predictions (many validation questions
also appear in the test set), so they are git-ignored. Please keep them private while the
competition is running.

## Notes

- **caw_geospeed_iq replaced caw_geospeed as the final entry, for the rules, not for the score.**
  The per-question plan was fixed before any validation or test scoring of caw_geospeed_iq and
  made it the last Track B upload whatever its score. On the rescored test questions it scored
  0.578 against caw_geospeed's 0.579. The notes below describe how caw_geospeed was chosen earlier.
- **caw_geospeed was kept as the final entry after seeing test scores.** Before the validation
  run, a plan fixed the variants (caw, caw_geospeed and the report-only caw_authors) and the
  decision rules: a validation gate (caw at least 0.5007, just above hybrid_v4's 0.5006) for
  running the test set; bands against hybrid_v4's
  test score of 0.552 (0.562 or more: clearly better; 0.543 to 0.561: cannot tell; 0.542 or
  less: worse); and that the last Track B upload would be whichever of hybrid_v4, caw and
  caw_geospeed scores highest on the test set. Picking the highest of three test scores is still
  a choice made after seeing them, so 0.577 is an optimistic estimate. The site reports only the
  overall and per-category scores, so there is no confidence interval for these differences.
- **caw alone did not beat hybrid_v4 on the test set:** 0.544 against 0.552 (-0.008, in the
  "cannot tell" band), with S2 its weakest category (0.479). On validation, caw had scored
  0.5518 against hybrid_v4's 0.5006 (+0.0512, 95% interval [-0.0334, +0.1477] from a paired,
  video-level bootstrap with 10,000 resamples).
- **The speed rule of caw_geospeed came from validation.** It follows an earlier validation
  analysis in which geometry scored 0.564 on speed questions against Qwen3-VL-8B's 0.336, so
  caw_geospeed's validation score (0.5842) was reported only and not used as evidence. On the
  test set, caw_geospeed scored +0.033 over caw (the only difference is the 708 speed questions;
  S2 went from 0.479 to 0.545) and +0.025 over hybrid_v4.
- **Most answers come from a third-party open fine-tune.** caw takes 3,287 of 3,289 answers and
  caw_geospeed 2,581 from Code-as-World-VL-9B, trained by its authors; this repository contributes
  the port, the fallback and the geometry answers on speed questions. The authors' paper reports
  0.554 for the 9B model on the QuantiPhy validation set (arXiv:2608.27549, Table 1) and does not
  say how the checkpoint was selected. Validation is weak evidence here: its 159 questions come
  from 24 videos that all appear in the test set (124 of the questions with identical text), its
  answers are public, and it had been used many times before.
- **Port fidelity.** The port differs from the authors' vLLM/Linux run (see
  [Method](#code-as-world-variants-caw-and-caw_geospeed)), so individual replies can differ from
  theirs. On validation, the port read with the authors' parser (caw_authors) scored 0.5518
  against the paper's 0.554 (by category -0.031, -0.007, +0.030 and -0.000 for S2, D2, S3, D3).
  The paper does not state how many validation questions it scored, and the authors' instructions
  use a validation CSV from the QuantiPhy GitHub repository that has since been removed there;
  this repository uses the Hugging Face copy, and the two files were not compared.
  `parse_answer_sci` and the authors' parser read the same number from every validation and test
  reply, so caw equals caw_authors on validation, and on the test set they differ only on the 2
  replies of 0.
- **Validation did not show hybrid_v4 to be better than hybrid_v2.** The hybrid_v4 rule was
  committed before the validation set was scored, together with a decision rule: adopt
  hybrid_v4 only if the lower end of the 95% interval of the difference is above 0, and stop
  this direction if the point estimate is below 0. On the 159 validation questions
  (24 videos), hybrid_v4 scored 0.5006 MRA and hybrid_v2 0.5044: a difference of -0.0038,
  95% interval [-0.0237, +0.0138] from a paired, video-level bootstrap (10,000 resamples).
  By that rule hybrid_v4 was not adopted and the direction was stopped. On validation,
  hybrid_v4 can differ from hybrid_v2 only on the 30 questions that hybrid_v2 sent to the VLM
  (26 of them took the 32B answer), so the interval is wide. Several geometry rules were added
  after looking at validation errors, so the validation scores of every version here are
  optimistic.
- **hybrid_v4 was chosen over hybrid_v2 after seeing official test-set scores.** Under the rule
  above, development stopped at hybrid_v2. hybrid_v4 became our Track B file (and later the
  fallback of caw and the reference of the caw gate) only because it had the highest official
  test-set score among the earlier open-weight uploads (+0.010 over hybrid_v2, higher in all four
  categories; with no confidence interval available, this difference may be noise):

  | Version | MRA | S2 | D2 | S3 | D3 |
  |---|---|---|---|---|---|
  | geometry_v1 | 0.516 | 0.507 | 0.522 | 0.563 | 0.472 |
  | hybrid_v1 (geometry, 8B answer where geometry fell back)* | 0.531 | 0.516 | 0.559 | 0.565 | 0.483 |
  | hybrid_v2 | 0.542 | 0.524 | 0.584 | 0.569 | 0.490 |
  | hybrid_v4 | 0.552 | 0.532 | 0.593 | 0.576 | 0.508 |

  \* hybrid_v1 is hybrid_v2 without the box check. Its original combine script is not part of
  this repository; `decide()` in `methods/hybrid_v2/hybrid.py` with `p_yes_threshold` set to 0
  applies the same rule and reproduces the uploaded values (differences only in the last
  floating-point digit).
- **No closed-API model is used anywhere in this repository.** The starter kit ships GPT-5.1
  outputs for the validation set (`external/QuantiPhy/model_outputs/gpt-5.1.csv`). They were
  used only as a reference score during development, and the tests use them as input to check
  our MRA scorer against the official evaluator and to exercise the bootstrap code. No pipeline
  step reads them.
- The fallback constants of geometry (medians of the test-set priors: 0.438 m, 1.4 m/s,
  9.7 m/s²) are computed from the prior text of the test questions only.

## Data and third-party material

- The QuantiPhy dataset cards (at the pinned revisions) release the annotations and metadata
  under CC BY 4.0; each video remains subject to its original license and terms of use. The
  data are **not** redistributed here; `scripts/download_data.py` fetches them from Hugging
  Face and the submission template from the competition site.
- `tests/` and this README quote a few question, prior and depth strings from the QuantiPhy
  datasets (CC BY 4.0) as parser fixtures and examples; no videos, answers or predictions are
  included.
- QuantiPhy: P. Li, T. Xiang, E. Mao, S. Wei, X. Chen, A. Masood, L. Fei-Fei, E. Adeli.
  *QuantiPhy: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of
  Vision-Language Models.* arXiv:2512.19526, 2025.

  ```bibtex
  @article{li2025quantiphy,
    title   = {QuantiPhy: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of Vision-Language Models},
    author  = {Li, Puyin and Xiang, Tiange and Mao, Ella and Wei, Shirley and Chen, Xinye and Masood, Adnan and Li, Fei-Fei and Adeli, Ehsan},
    journal = {arXiv preprint arXiv:2512.19526},
    year    = {2025}
  }
  ```
- The Qwen3-VL prompt reuses the system prompt and context wording of the official starter kit
  (MIT License, Copyright (c) 2025 Puyin Li).
- `methods/caw` re-implements the QuantiPhy recipe of the Code-as-World repository
  (<https://github.com/MirroS-Lab/Code-as-World>, commit `1353bf07d24e5463caff92ce23ffd34d03984831`,
  Apache License 2.0) for transformers; the authors' files are not included, and the files adapted
  from them carry a notice saying so. The model Code-as-World-VL-9B (Apache-2.0) is described in
  MirroS Team, *Code as Worlds: Agentic Discovery of Executable World Representations for Physical
  Reasoning*, arXiv:2608.27549, 2026.

  ```bibtex
  @article{mirros2026codeasworld,
    title   = {Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning},
    author  = {{MirroS Team}},
    journal = {arXiv preprint arXiv:2608.27549},
    year    = {2026}
  }
  ```
- Details and license texts: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are
  not included; see the model table above.

## License

The code in this repository is released under the MIT License; see [LICENSE](LICENSE). Portions of
`methods/caw/recipe.py`, `methods/caw/run.py` and `tests/test_caw.py` are adapted from
Code-as-World and remain under the Apache License 2.0, and the prompt wording taken from the
QuantiPhy starter kit remains under its MIT License (Copyright (c) 2025 Puyin Li). Both license
texts are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
