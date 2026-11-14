#!/bin/bash
# Reassemble the HashAttention signature weights (152 MB, split into <100 MB parts for GitHub) and check md5s.
cd "$(dirname "$0")"
cat assets/parts/llama3.1-8b-patch.64K.v1.hat_weights.pkl.part* > assets/llama3.1-8b-patch.64K.v1.hat_weights.pkl
md5sum -c - <<'M'
2bc4e3227ce18fd36b7d405cd1cedd58  assets/llama3.1-8b-patch.64K.v1.hat_weights.pkl
c393879f437c2fe6e5b9a45cf91a9b87  assets/Mstatic_lb_pre.pt
M
