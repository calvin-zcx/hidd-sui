mkdir output\log
mkdir output/log
python main_source.py --run_model KNN --knnk 50 --dump_detail --random_seed 50 2>&1 | tee output/log/knn_50.log
python main_source.py --run_model KNN --knnk 20 --dump_detail --random_seed 20 2>&1 | tee output/log/knn_20.log
python main_source.py --run_model KNN --knnk 10 --dump_detail --random_seed 10 2>&1 | tee output/log/knn_10.log
python main_source.py --run_model KNN --knnk 1 --dump_detail --random_seed 1 2>&1 | tee output/log/knn_1.log
#
# python main_source.py --run_model LR --dump_detail --random_seed 0 2>&1 | tee output/log/lr_0.log
# python main_source.py --run_model LR --random_seed 1 2>&1 | tee output/log/lr_1.log
# python main_source.py --run_model LR --random_seed 2 2>&1 | tee output/log/lr_2.log
# python main_source.py --run_model LR --random_seed 3 2>&1 | tee output/log/lr_3.log
# python main_source.py --run_model LR --random_seed 4 2>&1 | tee output/log/lr_4.log
# python main_source.py --run_model LR --random_seed 5 2>&1 | tee output/log/lr_5.log
# python main_source.py --run_model LIGHTGBM --dump_detail --random_seed 0 2>&1 | tee output/log/lightgbm_0.log
# python main_source.py --run_model LIGHTGBM --dump_detail --random_seed 0 2>&1 | tee output/log/lightgbm_0.log
# python main_source.py --run_model LIGHTGBM --random_seed 1 2>&1 | tee output/log/lightgbm_1.log
# python main_source.py --run_model LIGHTGBM --random_seed 2 2>&1 | tee output/log/lightgbm_2.log
# python main_source.py --run_model LIGHTGBM --random_seed 3 2>&1 | tee output/log/lightgbm_3.log
# python main_source.py --run_model LIGHTGBM --random_seed 4 2>&1 | tee output/log/lightgbm_4.log
# python main_source.py --run_model LIGHTGBM --random_seed 5 2>&1 | tee output/log/lightgbm_5.log