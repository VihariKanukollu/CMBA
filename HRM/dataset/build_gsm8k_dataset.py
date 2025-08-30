from typing import Optional, Tuple, List, Set
import os
import json
import re

import numpy as np
from datasets import load_dataset  # type: ignore
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    hf_name: str = "gsm8k"
    subset: str = "main"
    output_dir: str = "data/gsm8k-char-512"

    # Sequence shaping
    max_seq_len: int = 512

    # Text normalization
    lowercase: bool = True
    strip_spaces: bool = True

    # Optional controls
    subsample_size: Optional[int] = None


def _normalize_question(q: str, *, lowercase: bool, strip_spaces: bool) -> str:
    if lowercase:
        q = q.lower()
    # replace newlines/tabs with single spaces
    q = re.sub(r"[\t\n\r]+", " ", q)
    # collapse consecutive spaces
    if strip_spaces:
        q = re.sub(r"\s+", " ", q).strip()
    return q


def _extract_answer(ans: str) -> str:
    # GSM8K answers usually end with a line like: "#### 42"
    # Fallbacks: take the last number-like token if the pattern is missing.
    m = re.search(r"####\s*(.+)$", ans.strip().splitlines()[-1])
    if m:
        raw = m.group(1)
    else:
        # search in full text as a fallback
        m2 = re.findall(r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?)", ans)
        raw = m2[-1] if len(m2) else ""

    # Normalize: drop commas and spaces
    raw = raw.replace(",", "").strip()
    # Keep only characters relevant for numeric forms often seen in GSM8K
    keep = set("0123456789-./")
    normalized = "".join(ch for ch in raw if ch in keep)
    return normalized


def _build_charset(examples: List[Tuple[str, str]], *, lowercase: bool, strip_spaces: bool) -> List[str]:
    chars: Set[str] = set()
    for q, a in examples:
        qn = _normalize_question(q, lowercase=lowercase, strip_spaces=strip_spaces)
        chars.update(qn)
        # ensure all possible answer tokens are representable
        ans = _extract_answer(a)
        chars.update(ans)

    # Guarantee essential symbols exist
    for ch in "0123456789-./ ":
        chars.add(ch)

    # Remove the pad symbol if present (we reserve id 0 for PAD)
    # We don't need a special token; space is an actual character
    charset = sorted(chars)
    return charset


def _encode_question(q: str, char2id: np.ndarray, max_len: int, *, lowercase: bool, strip_spaces: bool) -> np.ndarray:
    qn = _normalize_question(q, lowercase=lowercase, strip_spaces=strip_spaces)
    # truncate if needed
    qn = qn[:max_len]
    arr = np.zeros((max_len,), dtype=np.uint8)
    # map chars to ids (1..V), PAD stays 0
    u8 = np.frombuffer(qn.encode("utf-8", errors="ignore"), dtype=np.uint8)
    # For multi-byte utf-8, fallback char-by-char
    if len(u8) != len(qn):
        for i, ch in enumerate(qn):
            if i >= max_len:
                break
            arr[i] = char2id[ord(ch) if ord(ch) < 256 else 0]
        return arr
    for i, ch in enumerate(qn):
        if i >= max_len:
            break
        arr[i] = char2id[ord(ch) if ord(ch) < 256 else 0]
    return arr


def _encode_answer_tail(ans_text: str, char2id: np.ndarray, max_len: int) -> np.ndarray:
    labels = np.zeros((max_len,), dtype=np.uint8)
    ans = _extract_answer(ans_text)
    if len(ans) == 0:
        return labels
    # If answer longer than seq len, keep the tail
    if len(ans) > max_len:
        ans = ans[-max_len:]
    # place answer right-aligned at the tail of the sequence
    start = max_len - len(ans)
    for i, ch in enumerate(ans):
        cid = char2id[ord(ch) if ord(ch) < 256 else 0]
        labels[start + i] = cid
    return labels


def _convert_split(hf_split, config: DataProcessConfig, charset: List[str]):
    # Build mapping table 0..255 -> id (0 is PAD)
    char2id = np.zeros(256, dtype=np.uint8)
    for idx, ch in enumerate(charset, start=1):
        if ord(ch) < 256:
            char2id[ord(ch)] = idx

    inputs: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    indices_group: List[int] = [0]
    indices_puzzle: List[int] = [0]
    puzzle_identifiers: List[int] = []

    example_id = 0
    puzzle_id = 0

    for ex in tqdm(hf_split):
        q: str = ex["question"]
        a: str = ex["answer"]

        inp = _encode_question(q, char2id, config.max_seq_len, lowercase=config.lowercase, strip_spaces=config.strip_spaces)
        lab = _encode_answer_tail(a, char2id, config.max_seq_len)

        inputs.append(inp)
        labels.append(lab)

        example_id += 1
        puzzle_id += 1

        indices_puzzle.append(example_id)
        puzzle_identifiers.append(0)
        indices_group.append(puzzle_id)

    # Stack
    inputs_np = np.stack(inputs, axis=0)
    labels_np = np.stack(labels, axis=0)

    results = {
        "inputs": inputs_np,
        "labels": labels_np,
        "group_indices": np.array(indices_group, dtype=np.int32),
        "puzzle_indices": np.array(indices_puzzle, dtype=np.int32),
        "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=config.max_seq_len,
        vocab_size=len(charset) + 1,  # PAD + charset
        pad_id=0,
        ignore_label_id=0,  # positions with 0 in labels are ignored
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(indices_group) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    return results, metadata


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    ds = load_dataset(config.hf_name, config.subset)

    # Optionally subsample training for quicker experiments
    train_data = list(ds["train"])  # type: ignore
    test_data = list(ds["test"])    # type: ignore

    if config.subsample_size is not None and config.subsample_size < len(train_data):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(train_data), size=config.subsample_size, replace=False)
        train_data = [train_data[i] for i in indices]

    # Build charset from both splits to avoid OOV at eval
    charset = _build_charset([(ex["question"], ex["answer"]) for ex in train_data + test_data],
                             lowercase=config.lowercase, strip_spaces=config.strip_spaces)

    # Save train/test
    for split_name, split_data in [("train", train_data), ("test", test_data)]:
        split_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        results, metadata = _convert_split(split_data, config, charset)

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)

        for k, v in results.items():
            np.save(os.path.join(split_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


if __name__ == "__main__":
    cli()


