#!/bin/bash
# Slurm job: the whole queue on one node, one job per GPU (8 lanes).
#   submit:  PY=/path/to/env/bin/python sbatch --partition=<PARTITION> --nodelist=<NODE> slurm/run_rest3pct.sbatch
#SBATCH --job-name=lc-rest3pct
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm/lc-rest3pct-%j.out
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PY=${PY:-python}
echo "host $(hostname)  job ${SLURM_JOB_ID:-?}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  PY=$PY"
$PY -c "import torch; print('torch', torch.__version__, 'gpus', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
GPUL=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" | tr ',' ' ')
echo "lanes on GPUs: $GPUL"
bash run_queue.sh queue.txt "$GPUL"
tar czf results_rest3pct.tgz results && echo "packed results_rest3pct.tgz"
