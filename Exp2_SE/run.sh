export CUDA_VISIBLE_DEVICES='0'

# pretrain f
python SE_pretraining.py --model-name 'BLSTM' --num-epochs 3
python SE_pretraining.py --model-name 'DDAE' --num-epochs 3

# adaptation: f --> g
python SE_finetuning_and_comparison.py --model-name 'BLSTM' --ntype "babycry" --SNR "-1"
python SE_finetuning_and_comparison.py --model-name 'DDAE' --ntype "babycry" --SNR "-1"