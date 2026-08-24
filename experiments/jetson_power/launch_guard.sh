#!/bin/bash
# Single-instance guard, then detach the campaign.
if pgrep -f "[r]un_catalog_ext.sh" > /dev/null; then
  echo "already running; refusing to double-launch"
  exit 1
fi
rm -f ~/aerobench/results_catalog/*__onnx_*.json
cd ~/aerobench
nohup ./run_catalog_ext.sh > catalog_ext.log 2>&1 < /dev/null &
echo "launched $!"
