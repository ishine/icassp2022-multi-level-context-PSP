import torch
from torch import nn


class MultiTaskLearningDecoder(nn.Module):
    def __init__(
        self,
        character_representation_dim,
        sentence_representation_dim,
        context_representation_dim,
        PW_decoder_hidden_dim,
        PPH_decoder_hidden_dim,
        IPH_decoder_hidden_dim,
    ):
        super(MultiTaskLearningDecoder, self).__init__()

        self.PW_decoder_hidden_dim = PW_decoder_hidden_dim
        self.PPH_decoder_hidden_dim = PPH_decoder_hidden_dim
        self.IPH_decoder_hidden_dim = IPH_decoder_hidden_dim
        self.multi_level_context_dim = (
            character_representation_dim
            + sentence_representation_dim
            + context_representation_dim
        )
        self.PW_decoder = nn.GRU(
            input_size=self.multi_level_context_dim + PW_decoder_hidden_dim,
            hidden_size=PW_decoder_hidden_dim,
            batch_first=True,
        )
        self.PPH_decoder = nn.GRU(
            input_size=self.multi_level_context_dim
            + PW_decoder_hidden_dim
            + PPH_decoder_hidden_dim,
            hidden_size=PPH_decoder_hidden_dim,
            batch_first=True,
        )
        self.IPH_decoder = nn.GRU(
            input_size=self.multi_level_context_dim
            + PW_decoder_hidden_dim
            + PPH_decoder_hidden_dim
            + IPH_decoder_hidden_dim,
            hidden_size=IPH_decoder_hidden_dim,
            batch_first=True,
        )
        self.PW_linear = nn.Linear(PW_decoder_hidden_dim, 2)
        self.PPH_linear = nn.Linear(PPH_decoder_hidden_dim, 2)
        self.IPH_linear = nn.Linear(IPH_decoder_hidden_dim, 2)

    def forward(
        self,
        batch_character_representation,
        batch_sentence_representation,
        batch_context_representation,
    ):
        # batch_character_representation: (batch_size,context_num,max_seq_len,character_representation_dim)
        # batch_sentence_representation: (batch_size,context_num,sentence_representation_dim)
        # batch_context_representation: (batch_size,context_num,context_representation_dim)
        (
            batch_size,
            context_num,
            max_seq_len,
            character_representation_dim,
        ) = batch_character_representation.shape
        batch_character_representation = batch_character_representation.reshape(
            (batch_size * context_num, max_seq_len, -1)
        )
        # batch_character_representation: (batch_size*context_num,max_seq_len,word_emb_dim)
        batch_sentence_representation = batch_sentence_representation.reshape(
            (batch_size * context_num, 1, -1)
        )
        # batch_sentence_representation: (batch_size*context_num,1,sentence_emb_dim)
        batch_context_representation = batch_context_representation.reshape(
            (batch_size * context_num, 1, -1)
        )
        # batch_context_representation: (batch_size*context_num,1,paragraph_emb_dim)
        PW_hidden_state = torch.zeros(
            (batch_size * context_num, 1, self.PW_decoder_hidden_dim),
            device=batch_character_representation.device,
        )
        PPH_hidden_state = torch.zeros(
            (batch_size * context_num, 1, self.PPH_decoder_hidden_dim),
            device=batch_character_representation.device,
        )
        IPH_hidden_state = torch.zeros(
            (batch_size * context_num, 1, self.IPH_decoder_hidden_dim),
            device=batch_character_representation.device,
        )
        PWs = []
        PPHs = []
        IPHs = []
        for i in range(max_seq_len):
            PW_hidden_state, _ = self.PW_decoder(
                torch.cat(
                    [
                        batch_character_representation[:, i].unsqueeze(1),
                        batch_sentence_representation,
                        batch_context_representation,
                        PW_hidden_state,
                    ],
                    dim=2,
                )
            )
            PPH_hidden_state, _ = self.PPH_decoder(
                torch.cat(
                    [
                        batch_character_representation[:, i].unsqueeze(1),
                        batch_sentence_representation,
                        batch_context_representation,
                        PW_hidden_state,
                        PPH_hidden_state,
                    ],
                    dim=2,
                )
            )
            IPH_hidden_state, _ = self.IPH_decoder(
                torch.cat(
                    [
                        batch_character_representation[:, i].unsqueeze(1),
                        batch_sentence_representation,
                        batch_context_representation,
                        PW_hidden_state,
                        PPH_hidden_state,
                        IPH_hidden_state,
                    ],
                    dim=2,
                )
            )

            PWs.append(PW_hidden_state)
            PPHs.append(PPH_hidden_state)
            IPHs.append(IPH_hidden_state)
        PW = torch.cat(PWs, dim=1).reshape((batch_size, context_num, max_seq_len, -1))
        PPH = torch.cat(PPHs, dim=1).reshape((batch_size, context_num, max_seq_len, -1))
        IPH = torch.cat(IPHs, dim=1).reshape((batch_size, context_num, max_seq_len, -1))
        return (
            self.PW_linear(PW).permute(0, 3, 1, 2),
            self.PPH_linear(PPH).permute(0, 3, 1, 2),
            self.IPH_linear(IPH).permute(0, 3, 1, 2),
        )
