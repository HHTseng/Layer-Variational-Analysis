export CUDA_VISIBLE_DEVICES='1'

scales='3 multiple'
model_name='SRCNN_02'
pretrain_epoch='100'
GD_finetune_epoch='50'
image_path='/home/htseng/Downloads/Transfer_Learning_datasets/preprocessing_SR_images/CUFED/CUFED_blurred_patches/'
test_image_path='/home/htseng/Downloads/Transfer_Learning_datasets/preprocessing_SR_images/Set14'
pretrained_model_dir='/home/htseng/Downloads/Transfer_Learning_Layer_Variational_Analysis_May5_BNL_local/Exp3_SRCNN/pretrained_models_SR/'
finetuned_model_dir='/home/htseng/Downloads/Transfer_Learning_Layer_Variational_Analysis_May5_BNL_local/Exp3_SRCNN/finetuned_models_SR/'

echo 'pretraining'
 for scale in $scales; do
     python SRCNN_pretraining.py --image-path $image_path \
                     --output-model-dir $pretrained_model_dir \
                     --model-name $model_name \
                     --scale $scale \
                     --lr 1e-4 \
                     --batch-size 128 \
                     --num-epochs $pretrain_epoch \
                     --num-workers 0 \
                     --seed 123
 done


N_samples='64 128 256 512 4096 8192'
pretrained_scale='3'
finetuned_scales='6 multiple'
  for finetuned_scale in $finetuned_scales; do
      for N in $N_samples; do
          echo 'GD finetuning'
          python SRCNN_GD_finetuning.py --image-path $image_path \
                      --model_name $model_name \
                      --pretrained-model-path $pretrained_model_dir \
                      --finetuned-model-path $finetuned_model_dir \
                      --pretrained-scale $pretrained_scale \
                      --finetuned-scale $finetuned_scale \
                      --batch-size 1 \
                      --num-epochs $GD_finetune_epoch \
                      --N-samples $N
          echo 'LVA finetuning'
          python SRCNN_LVA_comparisons.py --image-path $image_path\
                      --test-image-path $test_image_path\
                      --pretrained-model-path $pretrained_model_dir \
                      --finetuned-model-path $finetuned_model_dir \
                      --model-name $model_name \
                      --pretrained-scale $pretrained_scale \
                      --finetuned-scale $finetuned_scale \
                      --batch-size 1 \
                      --N-samples $N
      done
  done


pretrained_scale='multiple'
finetuned_scale='6'
  for N in $N_samples; do
     echo 'GD finetuning'
     python SRCNN_GD_finetuning.py --image-path $image_path\
                 --model_name $model_name \
                 --pretrained-model-path $pretrained_model_dir \
                 --finetuned-model-path $finetuned_model_dir \
                 --pretrained-scale $pretrained_scale \
                 --finetuned-scale $finetuned_scale \
                 --batch-size 1 \
                 --num-epochs $GD_finetune_epoch \
                 --N-samples $N
     echo 'LVA finetuning'
     python SRCNN_LVA_comparisons.py --image-path $image_path\
                 --test-image-path $test_image_path\
                 --pretrained-model-path $pretrained_model_dir \
                 --finetuned-model-path $finetuned_model_dir \
                 --model-name $model_name \
                 --pretrained-scale $pretrained_scale \
                 --finetuned-scale $finetuned_scale \
                 --batch-size 1 \
                 --N-samples $N
  done


