# Inference only on another scan with an already trained checkpoint
SCAN=data_scan0000000016
BASE=/pscratch/sd/s/skang2/4dcam
CKPT=$BASE/MgCl2_denoising/data_scan0000000035/experiment_XXXX/model/epoch=4.ckpt   # edit

python3 Utils/gaincorr_h5_to_tif.py --h5=$BASE/MgCl2_corr/${SCAN}_gaincorr.h5 --out_dir=$BASE/MgCl2_tif/

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
    --processor_num=32 \
    --precision=16 \
    --loss_function='L2' \
    --train=0 \
    --test=1 \
    --gpus=1 \
    --ckpt_path=$CKPT
