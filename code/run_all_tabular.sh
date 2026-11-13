cd "$(dirname "$0")"
mkdir -p ../results/logs
for cfg in "LGBM TX" "LR TX" "RF TX" "XGB TX" "LGBM TX+S" "LGBM TX+S+R" "RF TX+S+R" "XGB TX+S+R" "LGBM TX+W"; do
  set -- $cfg
  python3 run_tabular.py $1 $2 > ../results/logs/$1__$2.log 2>&1
done
