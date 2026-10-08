python3 ../main.py \
--common_path=../Experiment/Xingzhi_0144_denoising \
--training_path=../Datasets/Xingzhi/0144 \
--gt_path=../Datasets/Xingzhi/0144 \
--data_path_test=../Datasets/Xingzhi/0144 \
--save_folder_name=experiment \
--version_folder_name=5x5_blind_spot \
--model=5x5_blind  \
--img_size=256 \
--batch_size=16 \
--max_epochs=100 \
--recursive_factor=50 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--train=1 \
--test=1 \
--gpus=1 
#--ckpt_path=../Experiment/Xingzhi_0143_denoising/experiment20250829/model/epoch=28.ckpt