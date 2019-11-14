export VTT_DIR=.
export TASK_NAME=as

python ./run_answer_selection.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_eval \
    --do_lower_case \
    --data_dir $VTT_DIR/data \
    --max_seq_length 128 \
    --per_gpu_eval_batch=4 \
    --per_gpu_train_batch=4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --save_steps 1000 \
    --num_labels 2 \
    --output_dir $VTT_DIR/output_dir \
    --question_text "Who walks into the apartment?" \
    --clip_description_text "Chandler is sitting in a chair when Rachel walks into the apartment carrying two boxes." \
    --description_text "Chandler is sitting in a chair while Monica is sitting at the apartment." \
    --answer_candidates_list "Joey walks into the apartment." \
    --answer_candidates_list "Rachel walks into the apartment." \
