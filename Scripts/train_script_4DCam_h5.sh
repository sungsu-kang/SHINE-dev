
# NCEM 4D Camera workflow, starting from a *_gaincorr.h5 made by process_scan.py
SCAN=data_scan0000000008
BASE=/media/volume/data

# training + inference
python3 ../main.py \
    --common_path=$BASE/Experiment/$SCAN/ \
    --training_path=$BASE/Datasets/$SCAN/ \
    --data_path_test=$BASE/Datasets/$SCAN/ \
    --save_folder_name=experiment_1x1_defect \
    --version_folder_name=experiment \
    --model=defect_1x1_blind \
    --img_size=576 \
    --batch_size=4 \
    --filter=64 \
    --frame_num=21 \
    --max_epochs=1 \
    --processor_num=32 \
    --recursive_factor=1 \
    --learning_rate=0.0001 \
    --precision=16 \
    --loss_function='L2' \
    --train=1 \
    --test=1 \
    --gpus=1

# training + inference
python3 ../main.py \
    --common_path=$BASE/Experiment/$SCAN/ \
    --training_path=$BASE/Datasets/$SCAN/ \
    --data_path_test=$BASE/Datasets/$SCAN/ \
    --save_folder_name=experiment_3x3_defect \
    --version_folder_name=experiment \
    --model=defect_3x3_blind \
    --img_size=576 \
    --batch_size=4 \
    --filter=64 \
    --frame_num=21 \
    --max_epochs=1 \
    --processor_num=32 \
    --recursive_factor=1 \
    --learning_rate=0.0001 \
    --precision=16 \
    --loss_function='L2' \
    --train=1 \
    --test=1 \
    --gpus=1

# training + inference
python3 ../main.py \
    --common_path=$BASE/Experiment/$SCAN/ \
    --training_path=$BASE/Datasets/$SCAN/ \
    --data_path_test=$BASE/Datasets/$SCAN/ \
    --save_folder_name=experiment_5x5_defect\
    --version_folder_name=experiment \
    --model=defect_5x5_blind \
    --img_size=576 \
    --batch_size=4 \
    --filter=64 \
    --frame_num=21 \
    --max_epochs=1 \
    --processor_num=32 \
    --recursive_factor=1 \
    --learning_rate=0.0001 \
    --precision=16 \
    --loss_function='L2' \
    --train=1 \
    --test=1 \
    --gpus=1

# training + inference
python3 ../main.py \
    --common_path=$BASE/Experiment/$SCAN/ \
    --training_path=$BASE/Datasets/$SCAN/ \
    --data_path_test=$BASE/Datasets/$SCAN/ \
    --save_folder_name=experiment_7x7_defect \
    --version_folder_name=experiment \
    --model=defect_7x7_blind \
    --img_size=576 \
    --batch_size=4 \
    --filter=64 \
    --frame_num=21 \
    --max_epochs=1 \
    --processor_num=32 \
    --recursive_factor=1 \
    --learning_rate=0.0001 \
    --precision=16 \
    --loss_function='L2' \
    --train=1 \
    --test=1 \
    --gpus=1
