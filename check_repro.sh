#!/bin/bash
# Optional sanity check (~3-5 min): 16K dense qa_1 rows 100-102 vs the reference predictions.
cd "$(dirname "$0")"
GPU=${GPU:-0} NREQ=3 OUTROOT=check bash run_job.sh 16384 dense qa_1 || { echo "run failed, see check/logs/16384_dense_qa_1.txt"; exit 1; }
${PY:-python} - <<'P'
import glob, pandas as pd
ref = pd.read_csv("reference/len16384_dense_qa_1_rows100-102.csv", keep_default_na=False)["predicted_answer"].tolist()
f = glob.glob("check/rest3pct/len16384/dense_at_3pct/qa_1/raw_results.csv")[0]
new = pd.read_csv(f, keep_default_na=False)["predicted_answer"].head(3).tolist()
for i, (a, b) in enumerate(zip(ref, new)):
    print(f"[{i}] {'SAME' if a.strip() == b.strip() else 'DIFF'}  ref={a!r}  new={b!r}")
P
