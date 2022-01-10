import torch
from torch import nn
from .pretrained_language_model import PretrainedLanguageModel
from .multi_level_encoder import MultiLevelEncoder
from .multi_task_learning_decoder import MultiTaskLearningDecoder
from .positional_encoding import PositionalEncoding


class Model(nn.Module):
    def __init__(
        self,
        bert_feature_dim,
        PW_decoder_hidden_dim,
        PPH_decoder_hidden_dim,
        IPH_decoder_hidden_dim,
        max_length=128,
    ):
        super(Model, self).__init__()
        self.pretrained_language_model = PretrainedLanguageModel()
        self.positional_encoding = PositionalEncoding(
            bert_feature_dim, max_length=max_length
        )
        self.multi_level_encoder = MultiLevelEncoder(
            bert_feature_dim,
        )
        (
            _,
            sentence_representation_dim,
            context_representation_dim,
        ) = self.multi_level_encoder
        self.multi_task_learning_decoder = MultiTaskLearningDecoder(
            bert_feature_dim,
            sentence_representation_dim,
            context_representation_dim,
            PW_decoder_hidden_dim,
            PPH_decoder_hidden_dim,
            IPH_decoder_hidden_dim,
        )

    def forward(self, batch_input_ids, rank):
        batch_bert_feature, batch_attention_mask = self.pretrained_language_model(
            batch_input_ids, rank
        )
        # batch_bert_feature: (batch_size,context_num,max_seq_len,emb_dim)
        # batch_attention_mask: (batch_size,context_num,max_seq_len)
        (
            batch_size,
            context_num,
            max_seq_len,
            _,
        ) = batch_bert_feature.shape

        (
            batch_character_representation,
            batch_sentence_representation,
            batch_context_representation,
        ) = self.multi_level_encoder(
            self.positional_encoding(
                batch_bert_feature.reshape((batch_size * context_num, max_seq_len, -1))
            ).reshape((batch_size, context_num, max_seq_len, -1)),
            batch_attention_mask,
        )

        # batch_character_representation: (batch_size,context_num,max_seq_len,character_representation_dim)
        # batch_sentence_representation: (batch_size,context_num,sentence_representation_dim)
        # batch_context_representation: (batch_size,context_num,context_representation_dim)
        return self.decoder(
            batch_character_representation,
            batch_sentence_representation,
            batch_context_representation,
        )
