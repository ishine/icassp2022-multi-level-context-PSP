from transformers import AutoModel
import torch
from torch import nn

MODEL_NAME = "bert-base-chinese"
EMB_DIM = 768


class PretrainedLanguageModel(nn.Module):
    def __init__(self, finetune=False):
        super(PretrainedLanguageModel, self).__init__()
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, batch_input_ids, device):
        """
        batch_input_ids: list of input_ids dict from tokenizer
        """
        batch_output = []
        batch_attention_mask = []
        for x in batch_input_ids:
            input_ids = x["input_ids"].to(device)
            token_type_ids = x["token_type_ids"].to(device)
            attention_mask = x["attention_mask"].to(device)
            o = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            o = o.last_hidden_state.unsqueeze(0)
            batch_output.append(o)
            batch_attention_mask.append(attention_mask.unsqueeze(0))

        # return shape: (batch_size,context_num,max_length,bert_feature_dim), (batch_size,context_num,max_length)
        return (
            torch.cat(batch_output, dim=0),
            torch.cat(batch_attention_mask, dim=0) == 0,
        )
