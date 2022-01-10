import re
from transformers import AutoTokenizer
import os


def load_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)


def pred_to_text(
    texts,
    PW_labels,
    PPH_labels,
    IPH_labels,
):
    # text: (batch_size,context_num,length)
    # PW_labels: (batch_size,2,context_num,max_length)
    # PPH_labels: (batch_size,2,context_num,max_length)
    # IPH_labels: (batch_size,2,context_num,max_length)
    PW_labels = PW_labels.permute(0, 2, 3, 1).argmax(dim=-1)
    PPH_labels = PPH_labels.permute(0, 2, 3, 1).argmax(dim=-1)
    IPH_labels = IPH_labels.permute(0, 2, 3, 1).argmax(dim=-1)

    results = []
    for text, PW, PPH, IPH in zip(
        texts,
        PW_labels,
        PPH_labels,
        IPH_labels,
    ):
        context = []
        for line, line_PW, line_PPH, line_IPH in zip(
            text,
            PW,
            PPH,
            IPH,
        ):
            tokens = []
            for i in range(len(line)):
                # for i, t in enumerate(text):
                if line_IPH[i]:
                    tokens.append(text[i] + "#3")
                elif line_PPH[i]:
                    tokens.append(text[i] + "#2")
                elif line_PW[i]:
                    tokens.append(text[i] + "#1")
                else:
                    tokens.append(text[i])
            context.append("".join(tokens))
        results.append(context)
    return results
