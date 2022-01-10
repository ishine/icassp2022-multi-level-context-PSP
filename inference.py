import torch
from utils.dataloader import ProsodyStructurePredictionInferenceDataset
from utils.common import load_tokenizer, pred_to_text
from model.model import Model

from model.pretrained_language_model import EMB_DIM, MODEL_NAME
import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    # path to model checkpoint
    parser.add_argument("--model", type=str)
    # the window size of the context
    parser.add_argument("--context_num", type=int)
    parser.add_argument("--batch_size", type=int)
    # path to input file, one sentence per line
    parser.add_argument("--input", type=str)
    # gpu device for inference
    parser.add_argument("--device", type=int, default=0)
    # max input length
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


def inference(args):
    with open(args.model) as f:
        (
            _,
            PW_decoder_hidden_dim,
            PPH_decoder_hidden_dim,
            IPH_decoder_hidden_dim,
            encoder_state,
            decoder_state,
        ) = torch.load(f, map_location="cpu")

    model = Model(
        EMB_DIM,
        PW_decoder_hidden_dim,
        PPH_decoder_hidden_dim,
        IPH_decoder_hidden_dim,
        args.max_length,
    )
    model.multi_level_encoder.load_state_dict(encoder_state)
    model.multi_task_learning_decoder.load_state_dict(decoder_state)
    model = model.to(args.device)
    model.eval()
    dataset = ProsodyStructurePredictionInferenceDataset(
        args.input, args.context_num, load_tokenizer(MODEL_NAME), args.max_length
    )
    # labels here is useless,ignore
    for (text, input_ids, _, _, _, _, input_length) in dataset:
        PW, PPH, IPH = model(input_ids, args.device)
        for r in pred_to_text(text, PW, PPH, IPH):
            for line in r:
                print(line)


if __name__ == "__main__":
    inference(parse_args())
