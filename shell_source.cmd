mkdir output\log
mkdir output/log
python main_source.py --run_model LR --random_seed 0 2>&1 | tee output/log/lr_0.log
python main_source.py --run_model LR --random_seed 1 2>&1 | tee output/log/lr_1.log
python main_source.py --run_model LR --random_seed 2 2>&1 | tee output/log/lr_2.log
python main_source.py --run_model LR --random_seed 3 2>&1 | tee output/log/lr_3.log
python main_source.py --run_model LR --random_seed 4 2>&1 | tee output/log/lr_4.log
python main_source.py --run_model LR --random_seed 5 2>&1 | tee output/log/lr_5.log
python main_source.py --run_model LIGHTGBM --random_seed 0 2>&1 | tee output/log/lightgbm_0.log
python main_source.py --run_model LIGHTGBM --random_seed 1 2>&1 | tee output/log/lightgbm_1.log
python main_source.py --run_model LIGHTGBM --random_seed 2 2>&1 | tee output/log/lightgbm_2.log
python main_source.py --run_model LIGHTGBM --random_seed 3 2>&1 | tee output/log/lightgbm_3.log
python main_source.py --run_model LIGHTGBM --random_seed 4 2>&1 | tee output/log/lightgbm_4.log
python main_source.py --run_model LIGHTGBM --random_seed 5 2>&1 | tee output/log/lightgbm_5.log