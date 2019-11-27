# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np


class SequenceClassification(nn.Module):

    def __init__(self, dict_dim, embedding_dim, dropout_prob, num_labels, device):
        super(SequenceClassification, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.embedding = nn.Embedding(dict_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.device = device
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        encoder_norm = nn.LayerNorm(embedding_dim)
        self.q_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=encoder_norm)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        encoder_norm = nn.LayerNorm(embedding_dim)
        self.d_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=encoder_norm)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        encoder_norm = nn.LayerNorm(embedding_dim)
        self.sd_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=encoder_norm)

        self.q_d_Linear = nn.Linear(embedding_dim * 12, embedding_dim * 12)
        self.q_sd_Linear = nn.Linear(embedding_dim * 12, embedding_dim * 12)

        self.result_Linear = nn.Linear(embedding_dim * 24, embedding_dim * 24)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 48, embedding_dim * 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(embedding_dim * 32, embedding_dim * 8),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(embedding_dim * 8, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(embedding_dim * 4, num_labels)
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def similarity(self, context_vector, question_vector):
        '''
        :param context_vector: dims = [batch_size, max_sequence_length, 2d]
        :param question_vector: dims = [batch_size, max_sequence_length, 2d]
        :return: Similarity matrix. S , dims = [batch_size, max_sequence_length, max_sqeuence_langth]
        '''
        batch_size = context_vector.size(0)
        T = context_vector.size(1)  # max_sequence_length
        J = question_vector.size(1)  # max_sequence_length
        context_vector.to(self.device)
        context_vector_tiled = context_vector.unsqueeze(2)
        context_vector_tiled = context_vector_tiled.expand(-1, -1, J, -1)
        context_vector_tiled = context_vector_tiled.to(self.device)

        question_vector_tiled = question_vector.unsqueeze(1)
        question_vector_tiled = question_vector_tiled.expand(-1, T, -1, -1)
        question_vector_tiled = question_vector_tiled.to(self.device)

        hu = torch.cat((context_vector_tiled, question_vector_tiled, context_vector_tiled * question_vector_tiled), 3)
        # [batch,128,128,600]
        # create weight(dims=[1,1,6d]
        weights_size = self.embedding_dim * 6
        weights = torch.tensor((), dtype=torch.float32)
        weights = weights.to(self.device)
        weights = weights.new_ones((weights_size), dtype=torch.float32, requires_grad=True)
        weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        weights = self.dropout(weights)
        weights = weights.expand(batch_size, T, J, -1)  # [batch,128, 128, 600]

        assert hu.size() == weights.size()

        hu = hu.to(self.device)
        weights = weights.to(self.device)

        unsummed_dots = hu * weights  #
        similarity = unsummed_dots.sum(axis=3)

        return similarity

    def context_to_query(self, similarity, question_vector):
        """
        :param similarity: S vector, dims = [batch_size, max_sequence_length, max_sequence_length]
        :param question_vector: question vector, dims = [batch_size, max_sequence_length, embedding_size*2]
        :return: U_tilde vector, [batch_size, max_sequence_length, embedding_size*2](c2q)
        """
        softmax = nn.Softmax(dim=2)
        attention = softmax(similarity)
        attention = attention.to(self.device)

        T = similarity.size(1)
        question_vector_tiled = question_vector.unsqueeze(1)  # [batch, 1, max, 2d]
        question_vector_tiled = question_vector_tiled.expand(-1, T, -1, -1)  # [batch, max, max,2d]
        question_vector_tiled = question_vector_tiled.to(self.device)

        twod = question_vector.size(2)
        attention_tiled = attention.unsqueeze(3)  # [batch. max. max. 1]
        attention_tiled = attention_tiled.expand(-1, -1, -1, twod)  # [batch, max, max, 2d]
        attention_tiled = attention_tiled.to(self.device)

        aU = (attention_tiled * question_vector_tiled)

        U_tiled = aU.sum(axis=2)  # [batch, max, 2d]

        return U_tiled

    def query_to_context(self, similarity, context_vector):
        """
        :param similarity: S vector, dims = [batch_size, max_sequence_length, max_sequence_length]
        :param context_vector: context vector, dims = [batch_size, max_sequence_length, embedding_size*2]
        :return: H_tilde vector, [batch_size, max_sequence_length, embedding_size*2](q2c)
        """
        similarity_col_max = similarity.argmax(axis=2)  # [batch,seq_length,seq_length]->[batch,seq_length]
        similarity_col_max = similarity_col_max.type(dtype=torch.float32)
        similarity_col_max = similarity_col_max.to(self.device)
        softmax = nn.Softmax(dim=1)
        attention = softmax(similarity_col_max)

        # attention dim = [batch_size, max_sequence_length]

        twod = context_vector.size(2)
        attention_tiled = attention.unsqueeze(2)
        attention_tiled = attention_tiled.expand(-1, -1, twod)

        attention_tiled = attention_tiled.to(self.device)
        context_vector = context_vector.to(self.device)

        bH = attention_tiled * context_vector
        bH = bH.to(self.device)

        T = context_vector.size(1)
        h_tilde = bH.sum(axis=1)
        H_tilde = h_tilde.unsqueeze(1)
        H_tilde = H_tilde.expand(-1, T, -1)

        return H_tilde

    def final_attention(self, context_vector, c2q, q2c):
        """
        Compute HH_tilde, HU_tilde
        :param context_vector: description vector, dim = [batch, max_sequence_length, 2d]
        :param c2q: context to question vector, dim = [batch, max_sequence_length, 2d]
        :param q2c: question to context vector, dim = [batch, max_sequence_length, 8d]
        """
        HU_tiled = context_vector * c2q
        HH_tiled = context_vector * q2c

        qac = torch.cat((context_vector, c2q, HU_tiled, HH_tiled), 2)
        return qac

    def forward(self, q_vec, d_vec, sd_vec, labels=None):
        # embedding input vector
        q_emb = self.embedding(q_vec)
        d_emb = self.embedding(d_vec)
        sd_emb = self.embedding(sd_vec)

        q_transform = self.q_transformer(q_emb)
        d_transform = self.d_transformer(d_emb)
        sd_transform = self.sd_transformer(sd_emb)

        # Residual Net
        q_res = torch.cat((q_transform, q_emb),2)
        d_res = torch.cat((d_transform, d_emb),2)
        sd_res = torch.cat((sd_transform, sd_emb),2)

        # alignment
        q_d_similarity = self.similarity(d_res, q_res)
        d2q = self.context_to_query(q_d_similarity, d_res)
        q2d = self.query_to_context(q_d_similarity, q_res)
        q_d_final = self.final_attention(d_res, d2q, q2d)

        q_sd_similarity = self.similarity(sd_res, q_res)
        sd2q = self.context_to_query(q_sd_similarity, sd_res)
        q2sd = self.query_to_context(q_sd_similarity, q_res)
        q_sd_final = self.final_attention(sd_res, sd2q, q2sd)

        q_d_concat = torch.cat((q_res, q_d_final, d_res), 2)  # [batch_size, 128, embedding_size*12]
        q_sd_concat = torch.cat((q_res, q_sd_final, sd_res), 2)  # [batch_size, 128, embedding_size*12]

        q_d_linear = self.q_d_Linear(q_d_concat)  # [batch_size, 128, embedding_size*12]
        q_sd_linear = self.q_sd_Linear(q_sd_concat)  # [batch_size, 128, embedding_size*12]

        all_concat_input = torch.cat((q_d_linear, q_sd_linear), 2)  # [batch_size, 128, embedding_size*24]
        all_concat_input = all_concat_input.to(self.device)
        #all_concat_input = self.transformer_encoder(all_concat_input)
        result_output = self.result_Linear(all_concat_input)  # [batch_size, 128, embedding_size*24]
        result_output = result_output.to(self.device)

        result_output = result_output.permute(0, 2, 1)
        avg_pool = F.adaptive_avg_pool1d(result_output, 1)
        max_pool = F.adaptive_max_pool1d(result_output, 1)

        avg_pool = avg_pool.view(q_vec.size(0), -1)
        max_pool = max_pool.view(q_vec.size(0), -1)

        result = torch.cat((avg_pool, max_pool), 1)  # [batch_size, embedding_size*48]

        logits = self.classifier(result)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
