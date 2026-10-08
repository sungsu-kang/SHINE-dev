# NCEM 4D Camera workflow, starting from a *_gaincorr.h5 made by process_scan.py
SCAN=data_scan0000000035
BASE=/pscratch/sd/s/skang2/4dcam

# 1) gain-corrected h5 -> TIF folder (defective frames skipped)
python3 Utils/gaincorr_h5_to_tif.py \
    --h5=$BASE/MgCl2_corr/${SCAN}_gaincorr.h5 \
    --out_dir=$BASE/MgCl2_tif/

# 2) train + denoise
python3 main.py \
    --common_path=$BASE/MgCl2_denoising/$SCAN/ \
    --training_path=$BASE/MgCl2_tif/$SCAN/ \
    --data_path_test=$BASE/MgCl2_tif/$SCAN/ \
    --save_folder_name=experiment \
    --version_folder_name=MgCl2_1x1_defect \
    --model=defect_1x1_blind \
    --img_size=576 \
    --batch_size=4 \
    --filter=64 \
    --frame_num=5 \
    --max_epochs=5 \
    --processor_num=32 \
    --recursive_factor=1 \
    --learning_rate=0.0001 \
    --precision=16 \
    --loss_function='L2' \
    --test=1 \
    --gpus=1
