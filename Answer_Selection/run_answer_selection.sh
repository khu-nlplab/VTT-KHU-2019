export VTT_DIR=.
export TASK_NAME=as

python ./run_answer_selection.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $VTT_DIR/data \
    --max_seq_length 128 \
    --per_gpu_eval_batch=4 \
    --per_gpu_train_batch=4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --save_steps 1000 \
    --num_labels 5 \
    --output_dir $VTT_DIR/output_dir \

