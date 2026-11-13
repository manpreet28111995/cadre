cd "$(dirname "$0")"
while kill -0 337 2>/dev/null; do sleep 30; done
run() { python3 stream.py "$@" >> ../results/logs/stream.log 2>&1; }
for fs in TXS TXR; do
  for p in static every dpm; do run --model LGBM --fs $fs --policy $p --d 1 --seeds 0-4; done
done
echo done > ../results/logs/stream2_done.txt
