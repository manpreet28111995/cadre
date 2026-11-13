cd "$(dirname "$0")"
while [ ! -f ../results/logs/stream2_done.txt ]; do sleep 30; done
run() { python3 stream.py "$@" >> ../results/logs/stream.log 2>&1; }
for d in 1 3; do
  for p in static every dpm; do run --model LGBM --fs TXSR --policy $p --d $d --seeds 0-4 --nval 2 --tag _v2; done
done
echo done > ../results/logs/stream3_done.txt
