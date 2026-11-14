#!/bin/bash
# Download the 16K RULER dataset (public HF dataset xAlg-AI/att-hub-ruler-16k, 7 tasks) into the HF cache once,
# so the jobs can run with HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 (64K/96K data are already in data/ruler_local).
cd "$(dirname "$0")"
${PY:-python} - <<'P'
from benchmark.ruler16k import Ruler16K
df = Ruler16K(subsets_to_run=["qa_1","qa_2","vt","fwe","niah_multikey_2","niah_multikey_3","niah_multivalue"])._load_datasets()
print("16K rows:", len(df), df["task"].value_counts().to_dict() if "task" in df else "")
P
