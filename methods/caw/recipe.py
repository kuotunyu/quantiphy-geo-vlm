"""The authors' QuantiPhy recipe for Code-as-World-VL-9B: prompt text, answer parser, token layout.

Adapted from MirroS-Lab/Code-as-World (https://github.com/MirroS-Lab/Code-as-World, commit 1353bf07,
code_as_world/evaluation.py and code_as_world/templates/quantiphy_video.jinja), Apache License 2.0.
Changed from the original: the text-side parts (prompt text, answer parser, constants, short-clip
retry) are re-implemented as standalone functions that do not import vLLM. See
THIRD_PARTY_NOTICES.md.

Source of truth: external/Code-as-World/code_as_world/evaluation.py at commit AUTHORS_COMMIT
(README: `python -m code_as_world.evaluation 9b --input-csv ... --video-dir ...`). That module
imports vLLM at load time, so it cannot be imported here; each function below mirrors one of
theirs (named in its docstring) and tests/test_caw.py checks the outputs against their code on
real rows. Nothing in this file needs torch or transformers.

Token layout: the authors hand vLLM 0.19.1 the tokenized chat prompt, which contains one
"<|vision_start|><|video_pad|><|vision_end|>" triple. vLLM (Qwen3VLMultiModalProcessor,
PromptReplacement on that whole triple + get_video_repl) replaces it with, per temporal group of
2 frames, "<t seconds>" (tokenized on its own) + <|vision_start|> + <|video_pad|> * (H/32 * W/32)
+ <|vision_end|>. expand_video_tokens() rebuilds exactly that list of ids for Hugging Face
generate(); the transformers 5.11 processor text path would instead keep the outer
<|vision_start|>/<|vision_end|> around all frame blocks.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

from jinja2 import Template

AUTHORS_REPO = "https://github.com/MirroS-Lab/Code-as-World"
AUTHORS_COMMIT = "1353bf07d24e5463caff92ce23ffd34d03984831"
MODEL_ID = "MirroS-Lab/Code-as-World-VL-9B"  # Apache-2.0, base Qwen/Qwen3.5-9B
MODEL_REVISION = "46b111c53f5680eb12c63a7391e5c690d1e14ab1"

# evaluation.py L24-30 (the starter-kit zero-shot prompt joined into one line)
SYSTEM_PROMPT = (
    "You are an expert video analyst specializing in physics measurements. "
    "Analyze the video frames carefully and provide ONLY the numerical answer with units. "
    "No explanation or reasoning needed. Format your response as: [value] [unit]. "
    "Example: 2.5 cm. Be as accurate as possible with measurements and calculations. "
    "Please give me an estimated answer even if you are not sure."
)
# evaluation.py L69-72
DEPTH_PREFIX = (
    "Additionally, you have the following information about the distance between the objects "
    "in the video and the shooting camera:"
)
# templates/quantiphy_video.jinja after .strip() (evaluation.py L570)
FORMAT_PROMPT = (
    "<video> {{ content | trim }}\n\n"
    "Please answer the question with numbers and units ONLY. No explanation needed."
)
# templates/qwen3_5_no_think.jinja; byte-identical to the checkpoint's chat_template.jinja at MODEL_REVISION
CHAT_TEMPLATE_SHA256 = "22e67fd2f9b2fc41a36bc509c7aa87a96963dd76e7eb84977d907672d540355b"

# evaluation.py L37-57
MAX_PROMPT_LENGTH = 4096
MAX_RESPONSE_LENGTH = 512
MAX_MODEL_LEN = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH  # vLLM max_model_len (the prompt limit is on text ids)
VIDEO_NFRAMES = 16
VIDEO_TIMESTAMP_FPS = 24.0
VIDEO_FPS = 2.0  # passed as "video_fps", a key qwen-vl-utils never reads ("nframes" decides)
MIN_PIXELS = 0
MAX_PIXELS = 262144
SEED = 1
SAMPLING_CONFIG = {
    "temperature": 0.01, "top_p": 0.001, "top_k": -1, "min_p": 0.0, "presence_penalty": 0.0,
    "repetition_penalty": 1.0, "max_tokens": MAX_RESPONSE_LENGTH, "n": 1,
}

NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")  # L60
NFRAMES_INTERVAL_PATTERN = re.compile(r"nframes should in interval \[(\d+), (\d+)\], but got (\d+)")  # L61-63
_GIVEN_THAT_RE_TEMPLATE = r"^\s*Given\s+that\s+{}\s*[,.;:]\s*"  # L68

# config.json / tokenizer at MODEL_REVISION
IMAGE_TOKEN_ID = 248056  # <|image_pad|>
VIDEO_TOKEN_ID = 248057  # <|video_pad|>
VISION_START_ID = 248053  # <|vision_start|>
VISION_END_ID = 248054  # <|vision_end|>
EOS_TOKEN_ID = 248046  # <|im_end|>, the only stop id (generation_config.json)
PAD_TOKEN_ID = 248044  # <|endoftext|>
TEMPORAL_PATCH_SIZE = 2
SPATIAL_MERGE_SIZE = 2
LOGIT_BIAS = {IMAGE_TOKEN_ID: -100.0, VIDEO_TOKEN_ID: -100.0}  # evaluation.py _logit_bias (L457-466)

_FORMAT_TEMPLATE = Template(FORMAT_PROMPT)


def clean_text(value: Any) -> str:
    """_clean_text (L75-77): None / 'nan' / 'none' -> ''."""
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def positive_float(value: Any) -> float | None:
    """_positive_float (L84-89)."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def category(inference_type: Any, video_type: Any) -> str:
    """_category_from_fields (L92-98): 'S2' | 'D2' | 'S3' | 'D3' | ''."""
    inference = clean_text(inference_type).upper()
    video = clean_text(video_type).upper()
    if len(inference) < 1 or len(video) < 2:
        return ""
    prefix, dimension = inference[0], video[1]
    return f"{prefix}{dimension}" if prefix in {"S", "D"} and dimension in {"2", "3"} else ""


def record_from_row(row) -> dict[str, Any]:
    """The fields of the authors' CSV record (_load_csv L123-222) from a quantiphy.data loader row.

    The loader's `prior` is the CSV column ground_truth_prior. No answer field: the validation
    answer column is dropped before this is ever called.
    """
    question = clean_text(row.question)
    inference_type, video_type = clean_text(row.inference_type), clean_text(row.video_type)
    return {
        "video": str(row.video_path), "question": question, "raw_question": question,
        "video_id": clean_text(row.video_id), "video_source": clean_text(row.video_source),
        "video_type": video_type, "fps": clean_text(row.fps), "inference_type": inference_type,
        "ground_truth_prior": clean_text(row.prior), "depth_info": clean_text(row.depth_info),
        "category": category(inference_type, video_type),
    }


def normalise_question(value: Any) -> str:
    """_normalise_question (L237-239)."""
    question = clean_text(value).replace("？", "?")
    return question if not question or question[-1] in ".!?" else question + "?"


def content_from_record(record: dict[str, Any]) -> str:
    """_content_from_record (L242-264): 'Given that <prior>. [depth] <question>'."""
    existing = clean_text(record.get("content"))
    if existing:
        return existing
    prior = clean_text(record.get("ground_truth_prior"))
    question = clean_text(record.get("raw_question") or record.get("question"))
    if prior and question:
        pattern = _GIVEN_THAT_RE_TEMPLATE.format(re.escape(prior))
        stripped = re.sub(pattern, "", question, count=1, flags=re.IGNORECASE).strip()
        if stripped != question and stripped[:1].islower():
            stripped = stripped[:1].upper() + stripped[1:]
        question = stripped
    question = normalise_question(question)
    parts: list[str] = []
    if prior:
        parts.append(f"Given that {prior}.")
    depth = clean_text(record.get("depth_info"))
    if depth:
        parts.append(f"{DEPTH_PREFIX} {depth}")
    prefix = " ".join(parts).rstrip()
    if prefix and prefix[-1] not in ".!?":
        prefix += "."
    return f"{prefix} {question}".strip() if prefix else question


def format_prompt(record: dict[str, Any]) -> str:
    """_format_prompt (L267-268) with templates/quantiphy_video.jinja."""
    return _FORMAT_TEMPLATE.render(content=content_from_record(record))


def message_content(record: dict[str, Any]) -> list[dict[str, Any]]:
    """_message_content (L278-285): [{'type': 'video'}, {'type': 'text', 'text': ' Given that ...'}]."""
    content: list[dict[str, Any]] = []
    for index, text in enumerate(format_prompt(record).split("<video>")):
        if index:
            content.append({"type": "video"})
        if text:
            content.append({"type": "text", "text": text})
    return content


def build_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The messages of _build_prompt_ids (L302-315): system string + [video, text] user turn."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message_content(record)},
    ]


def strip_answer_tags(value: Any) -> str:
    """_strip_answer_tags (L469-474)."""
    text = str(value or "")
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def parse_prediction(value: Any) -> float | None:
    """_parse_prediction (L477-484): first number of the reply, no unit handling; None if none."""
    matches = NUMBER_PATTERN.findall(strip_answer_tags(value))
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def format_number(value: float | None) -> str:
    """_format_number (L524-525): the authors' CSV cell."""
    return "" if value is None else format(value, ".15g")


def retry_nframes(message: str) -> int | None:
    """_process_video's retry (L334-346): the frame count to ask for after a too-short-clip
    ValueError, or None when the authors would re-raise."""
    match = NFRAMES_INTERVAL_PATTERN.search(message)
    if match is None:
        return None
    min_allowed, max_allowed, requested = (int(v) for v in match.groups())
    if requested <= max_allowed or max_allowed < min_allowed:
        return None
    return max_allowed


def calculate_timestamps(indices: list[int], video_fps: float, merge_size: int = TEMPORAL_PATCH_SIZE) -> list[float]:
    """Qwen3VLProcessingInfo._calculate_timestamps (vLLM 0.19.1) == Qwen3VLProcessor._calculate_timestamps
    (transformers 5.11): pad to a multiple of merge_size with the last index, then the mean time of
    the first and last frame of each temporal group."""
    indices = list(indices)
    if len(indices) % merge_size != 0:
        indices = indices + [indices[-1]] * (merge_size - len(indices) % merge_size)
    timestamps = [idx / video_fps for idx in indices]
    return [(timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)]


def timestamp_text(t: float) -> str:
    return f"<{t:.1f} seconds>"


def expand_video_tokens(prompt_ids: list[int], grid_thw: tuple[int, int, int], timestamps: list[float],
                        encode: Callable[[str], list[int]]) -> list[int]:
    """Replace the single <|vision_start|><|video_pad|><|vision_end|> triple the way vLLM 0.19.1 does
    (Qwen3VLMultiModalProcessor.get_video_repl, no EVS). `encode(text)` must be
    tokenizer.encode(text, add_special_tokens=False)."""
    triple = [VISION_START_ID, VIDEO_TOKEN_ID, VISION_END_ID]
    hits = [i for i in range(len(prompt_ids) - 2) if prompt_ids[i:i + 3] == triple]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one video placeholder triple, found {len(hits)}")
    grid_t, grid_h, grid_w = (int(x) for x in grid_thw)
    if len(timestamps) != grid_t:
        raise ValueError(f"{len(timestamps)} timestamps for {grid_t} temporal groups")
    per_frame = grid_h * grid_w // (SPATIAL_MERGE_SIZE ** 2)
    repl: list[int] = []
    for t in timestamps:
        repl += list(encode(timestamp_text(t)))
        repl += [VISION_START_ID] + [VIDEO_TOKEN_ID] * per_frame + [VISION_END_ID]
    i = hits[0]
    return prompt_ids[:i] + repl + prompt_ids[i + 3:]


def mm_token_type_ids(ids: list[int]) -> list[int]:
    """Qwen3.5 get_rope_index input: 2 for video tokens, 1 for image tokens, 0 for text."""
    return [2 if t == VIDEO_TOKEN_ID else 1 if t == IMAGE_TOKEN_ID else 0 for t in ids]


def check_layout(ids: list[int], grid_thw: tuple[int, int, int]) -> None:
    """The invariants of the vLLM layout (and vLLM's max_model_len); raises ValueError.

    vLLM refuses a prompt longer than max_model_len and otherwise stops generating when prompt +
    output reach max_model_len (see max_new_tokens). A prompt of exactly max_model_len tokens leaves
    no room for an answer and is refused here too. No test prompt comes close (about 1.5-2.3k tokens)."""
    grid_t, grid_h, grid_w = (int(x) for x in grid_thw)
    n_video = sum(1 for t in ids if t == VIDEO_TOKEN_ID)
    if n_video != grid_t * grid_h * grid_w // SPATIAL_MERGE_SIZE ** 2:
        raise ValueError(f"{n_video} video tokens for grid {grid_thw}")
    if sum(1 for t in ids if t == VISION_START_ID) != grid_t or sum(1 for t in ids if t == VISION_END_ID) != grid_t:
        raise ValueError("vision start/end tokens do not match the temporal groups")
    if IMAGE_TOKEN_ID in ids:
        raise ValueError("unexpected image token")
    if len(ids) >= MAX_MODEL_LEN:
        raise ValueError(f"{len(ids)} prompt tokens: no room for an answer within vLLM's max_model_len {MAX_MODEL_LEN}")


def max_new_tokens(n_prompt: int) -> int:
    """vLLM 0.19.1 (v1 scheduler, check_stop) ends a request when it reaches max_tokens (512) output
    tokens or max_model_len (4608) tokens in total, whichever comes first."""
    return min(MAX_RESPONSE_LENGTH, MAX_MODEL_LEN - n_prompt)


def usable(value: float) -> bool:
    """The repo rule for a usable direct answer: a finite, non-zero number."""
    return value is not None and math.isfinite(value) and value != 0
