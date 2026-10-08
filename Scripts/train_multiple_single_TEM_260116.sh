
python3 ../main.py \
--file_type='single_dm4' \
--common_path=../Experiment/single-denoising-test \
--training_path=../Datasets/single-image-test \
--patches_folder=../Datasets/single-image-test-patches \
--data_path_test=../Datasets/single-image-test \
--patch_ratio=1 \
--patch_size=512 \
--patch_stride=256 \
--save_folder_name=experiment \
--version_folder_name=1x1_blind_spot \
--model=1x1_blind \
--img_size=256 \
--frame_num=1 \
--batch_size=8 \
--max_epochs=200 \
--learning_rate=0.001 \
--loss_function='L2' \
--precision=16 \
--recursive_factor=1 \
--processor_num=48 \
--prepare_patch=1 \
--train=1 \
--test=1 
#--ckpt_path=../Experiment/single-denoising-test/experiment20260116/model/epoch=111.ckpt