cd "$(dirname "$0")"
mkdir -p ../results/logs
run() { python3 stream.py "$@" >> ../results/logs/stream.log 2>&1; }
for d in 1 3 2 5; do
  for p in static every dpm slide per3 per5 psi oracle; do run --model LGBM --fs TXSR --policy $p --d $d --seeds 0-4; done
done
run --model LGBM --fs TXSR --policy static --d 1 --seeds 0-0 --diag --tag _diag
for d in 1 3 5; do
  for p in static every dpm; do run --model LGBM --fs TX --policy $p --d $d --seeds 0-4; done
done
for m in XGB RF; do
  for p in static every dpm; do run --model $m --fs TXSR --policy $p --d 1 --seeds 0-2; done
done
for c in 24 29 39; do
  for p in static every; do run --model LGBM --fs TXSR --policy $p --d 1 --seeds 0-2 --cut $c; done
done
