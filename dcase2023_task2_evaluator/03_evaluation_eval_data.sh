# basic parameters
## directory containing team results.
cd dcase2023_task2_evaluator
teams_root_dir=./teams
result_dir=${teams_root_dir}_result
## what depth to search '--teams_root_dir' using glob.
dir_depth=2

# Parameters for exporting additional results.
out_all=False
additional_result_dir=${teams_root_dir}_additional_result

# if using filename is DCASE2023 baseline style, change parameters as necessary.
# example filename: 'anomaly_score_DCASE2023T2<machine type>_<section>_test_seed<seed><tag>_Eval.csv'
seed=13711
tag="_id(0_)"

echo "python dcase2023_task2_evaluator.py"
python dcase2023_task2_evaluator.py \
    --dir_depth=$dir_depth \
    --teams_root_dir=$teams_root_dir \
    --result_dir=$result_dir \
    --additional_result_dir=$additional_result_dir \
    --out_all=$out_all \
    --seed=$seed \
    -tag=$tag \

