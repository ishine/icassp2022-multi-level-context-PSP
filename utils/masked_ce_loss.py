import torch
from torch import nn


class MaskedCELoss(nn.Module):
    def __init__(self):
        super(MaskedCELoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, PW_pred, PW_true, PPH_pred, PPH_true, IPH_pred, IPH_true, mask):
        # mask: (batch_size,context_num,max_seq_len)
        # PW_pred: (batch_size,2,context_num,max_seq_len)
        # PW_true: (batch_size,context_num,max_seq_len)
        # PPH_pred: (batch_size,2,context_num,max_seq_len)
        # PPH_true: (batch_size,context_num,max_seq_len)
        # IPH_pred: (batch_size,2,context_num,max_seq_len)
        # IPH_true: (batch_size,context_num,max_seq_len)
        mask = mask == 1
        pred_mask = mask.unsqueeze(3)
        PW_pred = PW_pred.permute(0, 2, 3, 1).masked_select(pred_mask).reshape((-1, 2))
        PPH_pred = (
            PPH_pred.permute(0, 2, 3, 1).masked_select(pred_mask).reshape((-1, 2))
        )

        IPH_pred = (
            IPH_pred.permute(0, 2, 3, 1).masked_select(pred_mask).reshape((-1, 2))
        )
        PW_true = PW_true.masked_select(mask)
        PPH_true = PPH_true.masked_select(mask)
        IPH_true = IPH_true.masked_select(mask)

        return (
            self.loss_fn(PW_pred, PW_true),
            self.loss_fn(PPH_pred, PPH_true),
            self.loss_fn(IPH_pred, IPH_true),
        )
