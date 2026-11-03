"""Per training sample, count the 13-grams shared with the eval tasks.

The training data is train_server's submit-time record for the run behind the
submitted checkpoint (the path ends in the run id)."""
from __future__ import annotations

import io
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Any

import httpx
import tiktoken

N = 13  # tokens per shingle

# PostTrainBench's tokenization: decon's normalization, then cl100k BPE.
_PUNCT = str.maketrans({c: " " for c in "!'\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~“”‘’—"})
_ENC = tiktoken.get_encoding("cl100k_base")


def _ngrams(text: str) -> set[tuple[int, ...]]:
    toks = _ENC.encode(" " + re.sub(r"\s+", " ", text.lower().translate(_PUNCT)).strip())
    return {tuple(toks[i : i + N]) for i in range(len(toks) - N + 1)}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def training_data_contamination(trial: Path) -> bool:
    checkpoint = (trial / "artifacts/workspace/final-checkpoint.txt").read_text().strip()
    run_id = checkpoint.rsplit("/", 1)[-1]
    response = httpx.get(
        f"{os.environ['TRAIN_SERVER_URL']}/v1/runs/{run_id}/artifacts",
        headers={"Authorization": f"Bearer {os.environ['TRAIN_SERVER_VERIFIER_API_KEY']}"},
        timeout=60,
    )
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
        data = tar.extractfile(f"{run_id}/data.jsonl").read().decode("utf-8", "replace")
    (trial / "audit/data.jsonl").write_text(data)

    # test_data.json: the full SkillsBench set (the eval tasks are a subset of it)
    # -- instruction, solution script, and held-out tests per task.
    index: dict[tuple[int, ...], str] = {}
    tasks = json.loads((Path(__file__).parent / "test_data.json").read_text())
    for task in tasks:
        for gram in _ngrams("\n".join(_strings(task))):
            index.setdefault(gram, task["task_name"])

    lines = [line for line in data.splitlines() if line.strip()]
    overlaps = []
    for line_no, line in enumerate(lines, 1):
        shared = {gram: task for gram in _ngrams("\n".join(_strings(json.loads(line))))
                  if (task := index.get(gram))}
        if shared:
            overlaps.append({"sample": line_no, "shared_13grams": len(shared),
                             "tasks": sorted(set(shared.values()))})

    info = {"run_id": run_id, "samples": len(lines), "overlaps": overlaps}
    (trial / "audit/training_data_contamination.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )
    return True
