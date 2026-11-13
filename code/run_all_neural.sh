cd "$(dirname "$0")"
python3 run_neural.py MLP 60 10 > ../results/logs/MLP__TX.log 2>&1
python3 run_neural.py GCN 150 5 > ../results/logs/GCN__TX.log 2>&1
python3 run_neural.py SAGE 150 5 > ../results/logs/SAGE__TX.log 2>&1
python3 run_neural.py GAT 100 5 > ../results/logs/GAT__TX.log 2>&1
