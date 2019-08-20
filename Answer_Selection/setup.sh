#!/bin/bash
# GloVe pretrained embeddings 다운로드

pip install -r requirements.txt

wget  http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d ./data/glove
rm -rf glove.6B.zip
