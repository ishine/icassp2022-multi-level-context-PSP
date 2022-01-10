import torch
from torch import nn
from .sentence_encoder import SentenceEncoder
from .context_encoder import ContextEncoder

CHARACTER_ENCODER_HEAD = 4 
CHARACTER_ENCODER_LAYER = 2

class MultiLevelEncoder(nn.Module):
    def __init__(
        self,
        bert_feature_dim
    ):
        super(MultiLevelEncoder, self).__init__()
        self.bert_feature_dim = bert_feature_dim
        self.character_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                bert_feature_dim, CHARACTER_ENCODER_HEAD, batch_first=True
            ),
            CHARACTER_ENCODER_LAYER,
        )
        self.sentence_encoder = SentenceEncoder(bert_feature_dim)
        self.context_encoder = ContextEncoder(self.sentence_encoder.output_dim)


    def forward(self, batch_bert_feature, batch_attention_mask):
        # batch_bert_feature: (batch_size,context_num,max_seq_len,bert_feature_dim)
        # batch_attention_mask: (batch_size,context_num,max_seq_len)
        batch_size, context_num, max_seq_len, bert_feature_dim = batch_bert_feature.shape
        batch_character_representation = self.character_encoder(
            batch_bert_feature.reshape((batch_size * context_num, max_seq_len, -1)),
            src_key_padding_mask=batch_attention_mask.reshape(
                (batch_size * context_num, max_seq_len)
            ),
        )
        # batch_character_representation: (batch_size*context_num,max_seq_len,emb_dim)
        batch_sentence_representation = self.sentence_encoder(
            batch_character_representation.reshape((batch_size, context_num, max_seq_len, bert_feature_dim))
        )
        # batch_sentence_representation: (batch_size,context_num,sentence_representation_dim)
        batch_context_representation = (
            self.context_encoder(batch_sentence_representation)
            .unsqueeze(1)
            .repeat(1, context_num, 1)
        )
        # batch_context_representation: (batch_size,context_num,context_representation_dim)

        return (
            batch_character_representation.reshape((batch_size, context_num, max_seq_len, -1)),
            batch_sentence_representation,
            batch_context_representation,
        )
    @property
    def output_dim(self):
        return (self.bert_feature_dim,self.sentence_encoder.output_dim,self.context_encoder.output_dim)