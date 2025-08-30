from typing import Optional, List, Tuple
import os
import json

import numpy as np
from datasets import load_dataset  # type: ignore
from argdantic import ArgParser
from pydantic import BaseModel
import sentencepiece as spm  # type: ignore
from tqdm import tqdm

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    hf_name: str = "gsm8k"
    subset: str = "main"
    output_dir: str = "data/gsm8k-bpe-2048"

    # Tokenizer
    sp_model_prefix: str = "data/gsm8k_sp"
    vocab_size: int = 32000
    character_coverage: float = 1.0

    # Sequence
    max_seq_len: int = 2048

    # Supervision spans
    include_rationale: bool = True

    # Labeling mode: "bpe" uses BPE ids, "digits" maps labels to 0..15 (0 pad, 1..10 digits incl '.' and '-')
    labels_mode: str = "bpe"

    subsample_size: Optional[int] = None


def _prepare_corpus_for_sp(train_data: List[dict], test_data: List[dict], fname: str):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as f:
        for ex in train_data + test_data:
            q = ex["question"].replace("\n", " ")
            a = ex["answer"].replace("\n", " ")
            f.write(q + "\n")
            f.write(a + "\n")


def _train_or_load_sp(cfg: DataProcessConfig, train_data: List[dict], test_data: List[dict]) -> spm.SentencePieceProcessor:
    model_file = cfg.sp_model_prefix + ".model"
    if not os.path.exists(model_file):
        corpus_file = cfg.sp_model_prefix + ".txt"
        _prepare_corpus_for_sp(train_data, test_data, corpus_file)
        spm.SentencePieceTrainer.Train(
            input=corpus_file,
            model_prefix=cfg.sp_model_prefix,
            vocab_size=cfg.vocab_size,
            character_coverage=cfg.character_coverage,
            model_type="bpe"
        )
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp


def _encode_gsm8k_example(sp: spm.SentencePieceProcessor, ex: dict, max_len: int, include_rationale: bool, labels_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    q = ex["question"].strip()
    a = ex["answer"].strip()

    # By convention in GSM8K, the final line contains "#### <number>".
    lines = a.splitlines()
    final_line = lines[-1] if len(lines) else ""
    # Build a "rationale + final" string; if rationale disabled, keep only final line
    target_text = (a if include_rationale else final_line)

    ids_inp = sp.encode(q, out_type=int)
    ids_tgt = sp.encode(target_text, out_type=int)

    # Truncate from the front for inputs (keep last max_len tokens to retain question end),
    # and from the front for targets (keep last max_len so answer tail is present)
    if len(ids_inp) > max_len:
        ids_inp = ids_inp[-max_len:]
    if len(ids_tgt) > max_len:
        ids_tgt = ids_tgt[-max_len:]

    # Right-pad to max_len (PAD=0 after we offset by +1 so PAD stays 0)
    inp = np.zeros((max_len,), dtype=np.int32)
    lab = np.zeros((max_len,), dtype=np.int32)
    if ids_inp:
        inp[:len(ids_inp)] = np.array(ids_inp, dtype=np.int32)
    if ids_tgt:
        lab[-len(ids_tgt):] = np.array(ids_tgt, dtype=np.int32)

    # Inputs always BPE-shifted (+1) to keep PAD=0
    inp = inp + (inp > 0)

    # Labels: either BPE-shifted to match input vocab, or compressed to small digit vocabulary
    if labels_mode == "bpe":
        lab = lab + (lab > 0)
    else:
        # Build digit-level mapping: 0 pad, 1..10 for '0'..'9', 11 '.' , 12 '-' , 13 '/' , 14 ' ' , 15 '+'
        DIGITS = {str(i): i + 1 for i in range(10)}
        EXTRA = {'.': 11, '-': 12, '/': 13, ' ': 14, '+': 15}
        # Convert target_text tail region only
        lab[:] = 0
        tail = target_text[-max_len:]
        for i, ch in enumerate(tail):
            idx = max_len - len(tail) + i
            if ch in DIGITS:
                lab[idx] = DIGITS[ch]
            elif ch in EXTRA:
                lab[idx] = EXTRA[ch]
            else:
                lab[idx] = 0
    return inp.astype(np.int32), lab.astype(np.int32)


def _convert_split(cfg: DataProcessConfig, sp: spm.SentencePieceProcessor, split_data: List[dict]):
    inputs = []
    labels = []
    puzzle_indices = [0]
    group_indices = [0]
    puzzle_identifiers = []
    example_id = 0
    puzzle_id = 0
    for ex in tqdm(split_data):
        inp, lab = _encode_gsm8k_example(sp, ex, cfg.max_seq_len, cfg.include_rationale, cfg.labels_mode)
        inputs.append(inp)
        labels.append(lab)
        example_id += 1
        puzzle_id += 1
        puzzle_indices.append(example_id)
        puzzle_identifiers.append(0)
        group_indices.append(puzzle_id)

    results = {
        "inputs": np.stack(inputs, 0),
        "labels": np.stack(labels, 0),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "group_indices": np.array(group_indices, dtype=np.int32),
        "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
    }

    # Output vocab matches labeling mode
    out_vocab = (sp.get_piece_size() + 1) if cfg.labels_mode == "bpe" else 16

    metadata = PuzzleDatasetMetadata(
        seq_len=cfg.max_seq_len,
        vocab_size=sp.get_piece_size() + 1,  # PAD + input BPE vocab
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(group_indices) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )
    return results, metadata


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    ds = load_dataset(config.hf_name, config.subset)
    train_data = list(ds["train"])  # type: ignore
    test_data = list(ds["test"])  # type: ignore

    if config.subsample_size is not None and config.subsample_size < len(train_data):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_data), size=config.subsample_size, replace=False)
        train_data = [train_data[i] for i in idx]

    sp = _train_or_load_sp(config, train_data, test_data)

    for split_name, split_data in [("train", train_data), ("test", test_data)]:
        results, metadata = _convert_split(config, sp, split_data)
        out_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
        for k, v in results.items():
            np.save(os.path.join(out_dir, f"all__{k}.npy"), v)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


if __name__ == "__main__":
    cli()


