#!/bin/bash
# Extended catalog campaign (resumable): skips cells whose result file already
# exists, so a restart never re-measures valid cells. ORT pinned to 1.18.1 —
# 1.19.2 aborts on Xavier when a power mode takes CPU cores offline. 10W is
# reboot-gated and deliberately absent.
cd ~/aerobench
ONNX="onnx_dlv3plus_w8a8 onnx_dlv3plus_float onnx_segformer_w8a8 onnx_segformer_float onnx_ffnet40s_w8a8 onnx_ffnet40s_float"
RES="dlv3_fp32_512 dlv3_fp16_512 lraspp_fp32_768 lraspp_fp16_768"
run_mode () {
  local id=$1 label=$2 workloads=$3
  echo rajat | sudo -S nvpmodel -m "$id" 2>/dev/null
  sleep 20
  echo "=== $label ($(date +%H:%M:%S)) ==="
  for w in $workloads; do
    for r in 1 2 3; do
      if [ -f "results_catalog/${label}__${w}__rep${r}.json" ]; then
        echo "skip ${label}__${w}__rep${r} (exists)"
        continue
      fi
      TORCH_HOME=/images/aerobench/torch_home python3 measure_catalog.py "$w" "$label" "$r" 2>&1 | tail -1
    done
  done
}
run_mode 3 MODE_30W_ALL "$ONNX"
run_mode 4 MODE_30W_6CORE "$ONNX $RES"
run_mode 5 MODE_30W_4CORE "$ONNX $RES"
run_mode 6 MODE_30W_2CORE "$ONNX $RES"
run_mode 2 MODE_15W "$ONNX $RES"
run_mode 7 MODE_15W_DESKTOP "$ONNX $RES"
run_mode 0 MAXN "$ONNX $RES"
echo rajat | sudo -S nvpmodel -m 3 2>/dev/null
echo "=== DONE, restored MODE_30W_ALL ($(date +%H:%M:%S)) ==="
