set -e
cd "$(dirname "$0")"
python3 download.py
python3 prep.py
python3 delay_features.py 1 2 3 5
python3 audit.py
python3 split_stats.py
python3 protocol_inflation.py
bash run_all_tabular.sh
bash run_all_neural.sh
python3 evaluate.py
python3 stats.py
python3 late_analysis.py
bash run_stream.sh
bash run_stream2.sh
bash run_stream3.sh
python3 stream_eval.py
python3 stream_stats.py
python3 explain.py
python3 cost_stream.py
python3 make_numbers.py
python3 make_numbers2.py
python3 make_tables2.py
python3 fig2_data_audit.py
python3 fig3_per_step.py
python3 fig4_delay_cost.py
python3 fig5_monitors.py
python3 fig6_curves.py
