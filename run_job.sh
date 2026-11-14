#!/bin/bash
# One job of the 3% context-length sweep (RULER hard 7, Llama-3.1-8B-Instruct): rows 100..199 of one task.
# Same commands as the paper's length table (V32: rows 0..99), one model per GPU, no split, no chunking.
#   usage: bash run_job.sh CTX_LEN METHOD TASK        CTX_LEN 16384|65536|98304, METHOD dense|ours|HAT|vAtt|MP
#   env:   PY (python, default python)  GPU (default 0)  NREQ (default 100)  OUTROOT (default results)
#          MODEL_PATH: local folder of Llama-3.1-8B-Instruct (config.json + safetensors); default = HF id from the HF cache
set -u
[ $# -ge 3 ] || { echo "usage: bash run_job.sh CTX_LEN METHOD TASK"; exit 2; }
L=$1; M=$2; T=$3; NREQ=${NREQ:-100}; OUTROOT=${OUTROOT:-results}
cd "$(dirname "$0")"
PY=${PY:-python}
[ -f assets/llama3.1-8b-patch.64K.v1.hat_weights.pkl ] || { echo "missing HAT weights (run: bash get_assets.sh)"; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU:-0}
export HUB_HAT_WEIGHT_FILE=$PWD/assets/llama3.1-8b-patch.64K.v1.hat_weights.pkl RULER_LOCAL_DIR=$PWD/data/ruler_local
PART="--profile-dtype bf16 --profile-source static+question --profile-static-file assets/Mstatic_lb_pre.pt --profile-rank 16 --profile-rot-window 32"
SEL_ST="--selector static --static-file assets/Mstatic_lb_pre.pt --static-rank 32 --static-window 32 --static-source static+question --static-share-kv --static-bits 8"
O=$OUTROOT/rest3pct/len$L; mkdir -p $O $OUTROOT/logs; LG=$OUTROOT/logs/${L}_${M}_$T.txt
MOD=""; if [ -n "${MODEL_PATH:-}" ]; then MOD="--model $MODEL_PATH"; export HUB_MODEL_KEY=meta-llama/Llama-3.1-8B-Instruct; fi
C="$MOD --device cuda:0 --ctx-len $L --output-dir $O --no-reusable --request-offset 100 --max-requests $NREQ --subsets $T"
case $L in 16384) MPL=210;; 65536) MPL=300;; 98304) MPL=250;; *) echo "bad CTX_LEN"; exit 2;; esac
case $M in
  dense) A="--densities 0.03 --methods dense";;
  ours)  A="--densities 0.03 --methods UTA+fair $SEL_ST $PART --profile-C $((64 * L / 32768)) --expand-tokens 0 --log-density --density-sampling 0.02";;
  HAT)   A="--densities 0.03 --methods HAT";;
  vAtt)  A="--densities 0.03 --methods vAttention --vatt-split 0.5 --eps 0.3 --delta 0.3 --log-density --density-sampling 0.02";;
  MP)    A="--methods magicpig --densities 0.03 --lsh-k 10 --lsh-l $MPL --log-density --density-sampling 0.02";;
  *) echo "bad METHOD"; exit 2;;
esac
RN=""; [ $M = vAtt ] && RN="vAttention(HAT)@3pct"; [ $M = MP ] && RN="MagicPIG(K10L${MPL})@3pct"
echo "job $L $M $T on GPU ${CUDA_VISIBLE_DEVICES} $(date -u +%F_%T)" > $LG
if [ -n "$RN" ]; then EXP_NAME=lc3pct_${L}_${M}_$T $PY run_multibin_hat_bench.py $C $A --run-name "$RN" >> $LG 2>&1
else EXP_NAME=lc3pct_${L}_${M}_$T $PY run_multibin_hat_bench.py $C $A >> $LG 2>&1; fi
