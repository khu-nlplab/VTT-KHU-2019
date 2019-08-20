#! /bin/bash
python preprocess_data.py \
  --data_dir=./data \
  --data_file=FriendsQA.json \
  --make_vocab=False \
  --analyze_data=False \
  --build_dicts=True \
  --dataset_type=multiple \
  --num_ans=2 \
 # --limit_ans_num=None
