#!/bin/bash
# Orin catalog campaign @ current MAXN only (mode switches need reboots on Orin).
# Everything lives on the NVMe: root filesystem is at 0 bytes and must not be touched.
if pgrep -f "[m]easure_catalog.py" > /dev/null; then echo "already running"; exit 1; fi
cd /mnt/work/aerobench/scripts
export TMPDIR=/mnt/work/aerobench/tmp
export PYTHONPATH=/mnt/work/aerobench/pylibs
export TORCH_HOME=/mnt/work/aerobench/torch_home
export CKPT_DIR=/mnt/work/aerobench/sc2_ckpt
export QAIHUB_DIR=/mnt/work/aerobench/qaihub
OUT=/mnt/work/aerobench/results_catalog
ALL="lraspp_fp32 lraspp_fp16 dlv3_fp32 dlv3_fp16 maskrcnn es_b064 es_b512 ghnd_bq3 fcm_head dlv3_fp32_512 dlv3_fp16_512 lraspp_fp32_768 lraspp_fp16_768 onnx_dlv3plus_w8a8 onnx_dlv3plus_float onnx_segformer_w8a8 onnx_segformer_float onnx_ffnet40s_w8a8 onnx_ffnet40s_float"
echo "=== ORIN MAXN ($(date +%H:%M:%S)) ==="
for w in $ALL; do
  for r in 1 2 3; do
    if [ -f "$OUT/MAXN__${w}__rep${r}.json" ]; then echo "skip MAXN__${w}__rep${r}"; continue; fi
    python3 measure_catalog.py "$w" MAXN "$r" "$OUT" 2>&1 | tail -1
  done
done
echo "=== DONE ($(date +%H:%M:%S)) ==="
