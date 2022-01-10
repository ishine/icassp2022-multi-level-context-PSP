import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=150)       # training epoch
    parser.add_argument("--lr", type=float, default=1e-5)       # training learning rate
    parser.add_argument(
        "--max_length", type=int, default=128
    )                                                           # max input length for BERT tokenizer
    parser.add_argument(
        "--log_dir", type=str, default="log"
    )                                                           # directory to store tensorboard and checkpoints
    parser.add_argument(
        "--devices", type=int, default=1
    )                                                           # number of devices used for training
    parser.add_argument("--seed", type=int, default=123456)     # random number seed
    parser.add_argument(
        "--port", type=str, default="12345"
    )                                                           # port for distributed training
    parser.add_argument(
        "--context_num", type=int, default=8
    )                                                           # the window size of the context
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--PW_decoder_hidden_dim", type=int, default=128)
    parser.add_argument("--PPH_decoder_hidden_dim", type=int, default=128)
    parser.add_argument("--IPH_decoder_hidden_dim", type=int, default=128)
    parser.add_argument("--train_dataset", type=str)            # path to training dataset
    parser.add_argument("--dev_dataset", type=str)              # path to dev dataset
    return parser.parse_args()
