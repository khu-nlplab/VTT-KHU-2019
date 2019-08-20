#! /bin/bash
python predict.py \
  --model_dir="./model" \
  --data_dir="./data" \
  --is_load=True \
  --question_text="What was Rachel's fling and Ross talking about?" \
  --description_text="Rachel's fling comes in and makes everyone on edge. Phoebe talk to Ross and Chandler about her problems. Rachel's fling and Ross begin talking about her and what his intentions are. " \
  --answer_candidates_list='They were talking about his true intentions towards Rachel.' \
  --answer_candidates_list='They were talking about who was the hottest.'
