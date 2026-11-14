# RULER context-length sweep, 3% read — remaining jobs (Llama-3.1-8B-Instruct)

논문 길이 표(3% read, RULER hard 7과제)를 과제당 200개로 완성하기 위한 남은 잡 42개 실행 패키지입니다.
원래 실험은 앞 100개(rows 0–99)만 돌렸고, 여기서는 **rows 100–199**를 돌립니다 (`--request-offset 100 --max-requests 100`).
명령은 원래 실험과 같고, GPU 한 장에 잡 하나씩 (분할·청크 없음) 돌립니다.

- 모델: `meta-llama/Llama-3.1-8B-Instruct` (bf16)
- 과제: qa_1, qa_2, vt, fwe, niah_multikey_2, niah_multikey_3, niah_multivalue

## 남은 잡 (`queue.txt`, 긴 잡 먼저)

| 길이 | dense | HashAttention | vAttention | MagicPIG | ours |
|---|---|---|---|---|---|
| 16K | – | – | – | 7 (L210) | – |
| 64K | – | – | – | 5 (L300; vt, fwe, mk2, mk3, mv) | – |
| 96K | 7 | 7 | 7 | 7 (L250) | 2 (mk3, mv) |

합계 42잡. 참고 소요(H100 1장): 96K ours ~1 h/잡, 64K/96K 베이스라인 ~1–2 h/잡 (추정). 8장이면 대략 6–10 h.

## 실행 (Slurm)

```bash
git clone https://github.com/hyunwoo137/longcontext.git
cd longcontext
conda create -n lc python=3.11 -y && conda activate lc
pip install -r requirements_pinned.txt
bash get_assets.sh                  # HashAttention 가중치 조립 + md5 확인 (OK 2줄)
export HF_TOKEN=...                 # Llama-3.1-8B-Instruct 접근 승인된 토큰 (모델·16K 데이터셋 첫 다운로드)

# (선택) 재현 확인: 16K dense qa_1 rows 100-102를 기준 예측과 비교 (~3-5분)
GPU=0 bash check_repro.sh

# 본 실행: 노드 1개(8 GPU), GPU마다 잡 하나
PY=$(which python) sbatch --partition=<파티션> --nodelist=<노드> slurm/run_rest3pct.sbatch
```

Slurm 없이: `nohup bash run_queue.sh queue.txt "0 1 2 3 4 5 6 7" > /dev/null 2>&1 &`

- 진행: `results/logs/queue.log` (gpuN start/done), 잡 로그 `results/logs/<길이>_<방법>_<과제>.txt`
- 정지: `touch STOP` → 지금 잡까지 마치고 종료. 다시 실행하면 끝난 잡(metrics.json 있음)은 건너뜀.
- 잡 하나만: `GPU=0 bash run_job.sh 98304 HAT qa_1`

## 결과 회수

결과는 `results/rest3pct/len<길이>/<방법 폴더>/<과제>/`(metrics.json, raw_results.csv, config.json)에 쌓입니다.
Slurm 잡은 끝나면 `results_rest3pct.tgz`를 만듭니다 (수동: `tar czf results_rest3pct.tgz results`). 이 파일을 전달해 주세요.
앞 100개 결과와 합쳐 과제별 200개 평균을 냅니다.

## 구성

- `run_multibin_hat_bench.py`, `sparse_attention_hub/`, `benchmark/`: 실험 코드 (HashAttention 가중치 경로만 환경변수 `HUB_HAT_WEIGHT_FILE`로 받도록 변경)
- `assets/Mstatic_lb_pre.pt`: QGS static metric (ours의 selector·proxy)
- `assets/parts/`: HashAttention 공개 signature 가중치(`llama3.1-8b-patch.64K.v1`, xAlg-AI HashAttention-1.0)를 90MB 조각으로 나눈 것
- `data/ruler_local/{65536,98304}`: 64K·96K RULER 데이터 (NVIDIA RULER 스크립트로 생성). 16K는 HF 데이터셋 `xAlg-AI/att-hub-ruler-16k`를 자동으로 받음
- MagicPIG L(LSH 테이블 수)은 길이별로 원래 표와 같은 값: 16K L210, 64K L300, 96K L250 (K=10)
- `requirements_pinned.txt`: 기준 실행 환경 버전 (Python 3.11)
