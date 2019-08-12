export BERT_BASE_DIR=uncased_L-12_H-768_A-12

pip install tensorflow
pip install torch
pip install tqdm

wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
rm -rf uncased_L-12_H-768_A-12.zip

python convert_tf_checkpoint_to_pytorch.py \
	--tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
	--bert_config_file $BERT_BASE_DIR/bert_config.json \
	--pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin


