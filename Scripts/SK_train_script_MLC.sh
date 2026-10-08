# The very first denoising used a 5x5 blind spot

# Third datasets: convert to tif and no use of patches.
# 0107_Insitu10: 3x3
# 0213_Insitu3: 1x1
# 0213_Insitu6: 3x3
# 0221_Insitu1: 1x1
# 0228_Insitu5: 3x3
# 0228_Insitu10-4: 1x1
# --img_size=256 \
# --batch_size=16 \
# --max_epochs=100 \
# --recursive_factor=10 \
# --learning_rate=0.001 \
# --precision=16 \
# --loss_function='L2' \
# --train=1 \
# --test=1 

# Fourth dataset
# python3 ../main.py \
# --file_type='large_dm4' \
# --common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260513/InSitu7/HAADF_denoised/' \
# --training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260513/InSitu7/HAADF/' \
# --data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260513/InSitu7/HAADF/' \
# --patches_folder='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260513/InSitu7/HAADF_patches/' \
# --patch_ratio=0.5 \
# --patch_size=512 \
# --patch_stride=256 \
# --save_folder_name=experiment \
# --version_folder_name=3x3_blind_spot \
# --model=3x3_blind \
# --img_size=256 \
# --batch_size=16 \
# --max_epochs=100 \
# --recursive_factor=10 \
# --learning_rate=0.001 \
# --precision=16 \
# --loss_function='L2' \
# --prepare_patch=1 \
# --train=1 \
# --test=1

# Fifth Dataset
# python3 ../main.py \
# --file_type='large_dm4' \
# --common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260710/619_InSitu1/HAADF_denoised/' \
# --training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260710/619_InSitu1/HAADF/' \
# --data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260710/619_InSitu1/HAADF/' \
# --patches_folder='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260710/619_InSitu1/HAADF_patches/' \
# --patch_ratio=0.5 \
# --patch_size=512 \
# --patch_stride=256 \
# --save_folder_name=experiment \
# --version_folder_name=3x3_blind_spot \
# --model=3x3_blind \
# --img_size=256 \
# --batch_size=16 \
# --max_epochs=100 \
# --recursive_factor=10 \
# --learning_rate=0.001 \
# --precision=16 \
# --loss_function='L2' \
# --prepare_patch=1 \
# --train=1 \
# --test=1

# Sixth dataset
# python3 ../main.py \
# --common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260716/619_Insitu4/LAADF_denoised/' \
# --training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260716/619_Insitu4/LAADF-tif/' \
# --data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260716/619_Insitu4/LAADF-tif/' \
# --save_folder_name=experiment \
# --version_folder_name=1x1_blind_spot \
# --model=1x1_blind \
# --img_size=256 \
# --batch_size=8 \
# --max_epochs=100 \
# --recursive_factor=1 \
# --learning_rate=0.001 \
# --precision=16 \
# --loss_function='L2' \
# --train=1 \
# --test=1

python3 ../main.py \
--common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_731_S-1_Insitu(1)/HAADF_denoised/' \
--training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_731_S-1_Insitu(1)/HAADF-tif/' \
--data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_731_S-1_Insitu(1)/HAADF-tif/' \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind \
--img_size=256 \
--batch_size=8 \
--max_epochs=100 \
--recursive_factor=1 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--train=1 \
--test=1

python3 ../main.py \
--common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_731_S-1_Insitu(1)/LAADF_denoised/' \
--training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_731_S-1_Insitu(1)/LAADF-tif/' \
--data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_731_S-1_Insitu(1)/LAADF-tif/' \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind \
--img_size=256 \
--batch_size=8 \
--max_epochs=100 \
--recursive_factor=1 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--train=1 \
--test=1

python3 ../main.py \
--common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_731_S-1_Insitu(1)/ABF_denoised/' \
--training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_731_S-1_Insitu(1)/ABF-tif/' \
--data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_731_S-1_Insitu(1)/ABF-tif/' \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind \
--img_size=256 \
--batch_size=8 \
--max_epochs=100 \
--recursive_factor=1 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--train=1 \
--test=1

python3 ../main.py \
--common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_828_Insitu(9)/HAADF_denoised/' \
--training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_828_Insitu(9)/HAADF-tif/' \
--data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_828_Insitu(9)/HAADF-tif/' \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind \
--img_size=256 \
--batch_size=8 \
--max_epochs=100 \
--recursive_factor=1 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--train=1 \
--test=1

python3 ../main.py \
--common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_828_Insitu(9)/LAADF_denoised/' \
--training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_828_Insitu(9)/LAADF-tif/' \
--data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_828_Insitu(9)/LAADF-tif/' \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind \
--img_size=256 \
--batch_size=8 \
--max_epochs=100 \
--recursive_factor=1 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--train=1 \
--test=1

python3 ../main.py \
--common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_828_Insitu(10)/HAADF_denoised/' \
--training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_828_Insitu(10)/HAADF-tif/' \
--data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_828_Insitu(10)/HAADF-tif/' \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind \
--img_size=256 \
--batch_size=8 \
--max_epochs=100 \
--recursive_factor=1 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--train=1 \
--test=1

python3 ../main.py \
--common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_828_Insitu(10)/LAADF_denoised/' \
--training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_828_Insitu(10)/LAADF-tif/' \
--data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/monoC_828_Insitu(10)/LAADF-tif/' \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind \
--img_size=256 \
--batch_size=8 \
--max_epochs=100 \
--recursive_factor=1 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--train=1 \
--test=1

python3 ../main.py \
--common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(8)/HAADF_denoised/' \
--training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(8)/HAADF-tif/' \
--data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(8)/HAADF-tif/' \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind \
--img_size=256 \
--batch_size=8 \
--max_epochs=100 \
--recursive_factor=1 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--train=1 \
--test=1

python3 ../main.py \
--common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(8)/LAADF_denoised/' \
--training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(8)/LAADF-tif/' \
--data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(8)/LAADF-tif/' \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind \
--img_size=256 \
--batch_size=8 \
--max_epochs=100 \
--recursive_factor=1 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--train=1 \
--test=1

python3 ../main.py \
--common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(9)/HAADF_denoised/' \
--training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(9)/HAADF-tif/' \
--data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(9)/HAADF-tif/' \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind \
--img_size=256 \
--batch_size=8 \
--max_epochs=100 \
--recursive_factor=1 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--train=1 \
--test=1

python3 ../main.py \
--common_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(9)/LAADF_denoised/' \
--training_path='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(9)/LAADF-tif/' \
--data_path_test='/mnt/g/Sungsu/Denoising/Shoaib-MLC/20260919/710_80A_Insitu(9)/LAADF-tif/' \
--save_folder_name=experiment \
--version_folder_name=3x3_blind_spot \
--model=3x3_blind \
--img_size=256 \
--batch_size=8 \
--max_epochs=100 \
--recursive_factor=1 \
--learning_rate=0.001 \
--precision=16 \
--loss_function='L2' \
--train=1 \
--test=1