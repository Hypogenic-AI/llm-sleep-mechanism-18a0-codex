# Downloaded Datasets

This directory contains local datasets for the research project. Data files are excluded from git by `datasets/.gitignore`; documentation and small `samples/` files are retained.

## Summary

| Dataset | Source | Local Location | Size | Use |
|---|---|---:|---:|---|
| XSTest | `Paul/XSTest` | `datasets/xstest/` | 450 prompts | False-refusal and exaggerated-safety evaluation |
| Do-Not-Answer | `LibrAI/do-not-answer` | `datasets/do_not_answer/` | 939 prompts plus model responses | Safety/refusal action labels |
| OR-Bench 80K | `bench-llm/or-bench`, config `or-bench-80k` | `datasets/or_bench_80k/` | 80,359 prompts | Large over-refusal prompt set |
| OR-Bench Hard 1K | `bench-llm/or-bench`, config `or-bench-hard-1k` | `datasets/or_bench_hard_1k/` | 1,319 prompts | Hard over-refusal subset |
| OR-Bench Toxic | `bench-llm/or-bench`, config `or-bench-toxic` | `datasets/or_bench_toxic/` | 655 prompts | Toxic contrast set |
| PersonaMem-v2 Benchmark Text | `bowen-upenn/PersonaMem-v2`, config `benchmark`, split `benchmark_text` | `datasets/personamem_v2_benchmark_text/` | 5,000 rows | Long-context user preference/persona benchmark |
| WildChat bounded sample | `allenai/WildChat`, streaming train | `datasets/wildchat_sample/` | 200 sampled, 25 sleep-phrase matches from 14,782 scanned conversations | Real-world chat examples and phrase search |

Full dataset metadata is saved in `datasets/dataset_summary.json`.

## Loading Local Datasets

Use Hugging Face `datasets` for the saved Arrow datasets:

```python
from datasets import load_from_disk

xstest = load_from_disk("datasets/xstest")
or_hard = load_from_disk("datasets/or_bench_hard_1k")
personamem = load_from_disk("datasets/personamem_v2_benchmark_text")
```

WildChat samples are JSON files:

```python
import json
from pathlib import Path

matches = json.loads(Path("datasets/wildchat_sample/sleep_phrase_matches/sample.json").read_text())
```

## Re-Download Instructions

```python
from datasets import load_dataset

load_dataset("Paul/XSTest").save_to_disk("datasets/xstest")
load_dataset("LibrAI/do-not-answer").save_to_disk("datasets/do_not_answer")
load_dataset("bench-llm/or-bench", "or-bench-80k").save_to_disk("datasets/or_bench_80k")
load_dataset("bench-llm/or-bench", "or-bench-hard-1k").save_to_disk("datasets/or_bench_hard_1k")
load_dataset("bench-llm/or-bench", "or-bench-toxic").save_to_disk("datasets/or_bench_toxic")
load_dataset("bowen-upenn/PersonaMem-v2", "benchmark", split="benchmark_text").save_to_disk(
    "datasets/personamem_v2_benchmark_text"
)
```

For WildChat, use streaming to avoid downloading the full corpus:

```python
from datasets import load_dataset

wild = load_dataset("allenai/WildChat", split="train", streaming=True)
for i, row in enumerate(wild):
    if i >= 1000:
        break
```

## Gated or Not Fully Downloaded Sources

`sorry-bench/sorry-bench-202406`, `sorry-bench/sorry-bench-202503`, and `lmsys/lmsys-chat-1m` require Hugging Face authentication or access approval. The SORRY-Bench code repository is cloned under `code/sorry_bench/`, and the paper PDF is in `papers/`, but the gated dataset itself was not downloaded.

## Quick Validation Notes

- All saved Hugging Face datasets loaded successfully from disk.
- `datasets/wildchat_sample/sleep_phrase_matches/sample.json` contains 25 phrase matches. A quick inspection shows several are ordinary sleep-topic content or user sleep questions, not necessarily assistant-initiated bedtime nudges; downstream experiments should classify direction and context.
- Small sample files are available under each dataset's `samples/` directory.

