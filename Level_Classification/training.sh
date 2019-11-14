python  run_classifier.py \
	--data_dir SNUdataset \
	--task_name logic \
	--vocab_file vocab.txt \
	--output_dir result \
	--train_batch_size 8 \
	--eval_batch_size 8 \
	--do_lower_case \
	--max_seq_length 128 \
	--do_train \
	--do_eval \
	--learning_rate 1e-5 \
	--num_train_epochs 60.0 \
	--embedding_dim 200 \
	--local_rank -1 \
	#--load_model ./model/memory_model_15000.bin \
