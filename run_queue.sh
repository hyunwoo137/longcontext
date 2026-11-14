#!/bin/bash
# One lane per GPU; each lane takes the next unclaimed line ("CTX_LEN METHOD TASK") of the queue file.
# A job counts as done when its metrics.json exists (re-running the queue skips it).
#   usage: bash run_queue.sh [QUEUE_FILE] [GPU_LIST]      default: queue.txt "0 1 2 3 4 5 6 7"
#   stop after the current jobs: touch STOP
cd "$(dirname "$0")"; Q=${1:-queue.txt}; GPUL=${2:-"0 1 2 3 4 5 6 7"}; mkdir -p results/logs
CL=results/.claimed_$(basename $Q); : > $CL
log() { echo "$(date -u +%F_%T) $*" | tee -a results/logs/queue.log; }
done_job() { L=$1; M=$2; T=$3
  case $M in dense) N=dense_at_3pct;; HAT) N=HAT_at_3pct;; vAtt) N="vAttention(HAT)_at_3pct";; MP) N="MagicPIG(K10L*)_at_3pct";; ours) N="OursFair*_at_3pct*";; esac
  ls results/rest3pct/len$L/$N/$T/metrics.json >/dev/null 2>&1; }
lane() { G=$1
  while [ ! -f STOP ]; do
    j=$(flock results/.lock bash -c "while read -r l; do [ -z \"\$l\" ] && continue; grep -qxF \"\$l\" $CL && continue; echo \"\$l\" >> $CL; echo \"\$l\"; exit 0; done < $Q")
    [ -z "$j" ] && break
    done_job $j && { log "gpu$G skip $j (done)"; continue; }
    log "gpu$G start $j"; GPU=$G bash run_job.sh $j; rc=$?
    if [ $rc != 0 ] && done_job $j; then rc=0; fi
    log "gpu$G done $j rc=$rc"
  done; }
for G in $GPUL; do lane $G & sleep 5; done; wait
log "queue done"
