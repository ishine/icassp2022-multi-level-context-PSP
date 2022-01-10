import torch
from torch.utils.data import DataLoader
from utils.dataloader import ProsodyStructurePredictionDataset, DatasetCollateFn
from utils.common import load_tokenizer
from model.model import Model

from model.pretrained_language_model import EMB_DIM, MODEL_NAME
import os
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import train_args
from torch.optim import Adam
import random
from utils.masked_ce_loss import MaskedCELoss


def cal_metric(y_true, y_pred, input_length):
    # y_true: (batch_size,context_num,max_seq_length)
    # y_pred: (batch_size,2,context_num,max_seq_length)
    # input_length: (batch_size,context_num)
    y_true = y_true.cpu().detach()
    y_pred = y_pred.cpu().detach().argmax(dim=1)
    # y_pred: (batch_size,context_num,max_seq_length)
    y_true_truncated = torch.cat(
        [
            sentence[:sentence_length]
            for context, context_length in zip(y_true, input_length)
            for sentence, sentence_length in zip(context, context_length)
        ]
    )
    y_pred_truncated = torch.cat(
        [
            sentence[:sentence_length]
            for context, context_length in zip(y_pred, input_length)
            for sentence, sentence_length in zip(context, context_length)
        ]
    )
    y_true = y_true_truncated.numpy()
    y_pred = y_pred_truncated.numpy()
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),  # tp / (tp + fp)
        recall_score(y_true, y_pred, zero_division=0),  # tp / (tp + fn)
        f1_score(y_true, y_pred, zero_division=0),
    )


def main():
    args = train_args.parse_args()
    mp.spawn(train, args=(args), nprocs=args.devices, join=True)


def train(rank, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dist.init_process_group("nccl", rank=rank, world_size=args.devices)

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    loss_fn = MaskedCELoss()
    if rank == 0:
        summary = SummaryWriter(args.log_dir)

    tokenizer = load_tokenizer(MODEL_NAME)

    train_dataset = ProsodyStructurePredictionDataset(
        args.train_dataset, args.context_num, tokenizer, args.max_length
    )
    dev_dataset = ProsodyStructurePredictionDataset(
        args.dev_dataset, args.context_num, tokenizer, args.max_length
    )
    collate_fn = DatasetCollateFn()
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn
    )

    model = Model(
        EMB_DIM,
        args.PW_decoder_hidden_dim,
        args.PPH_decoder_hidden_dim,
        args.IPH_decoder_hidden_dim,
        args.max_length,
    )
    ddp_model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=True)

    updater = Adam(
        filter(lambda x: x.requires_grad, ddp_model.parameters()), lr=args.lr
    )
    train_step = 1
    dev_step = 1

    for i in range(args.epoch):
        ddp_model.train()
        for (
            batch_text,
            batch_input_ids,
            batch_PW_labels,
            batch_PPH_labels,
            batch_IPH_labels,
            batch_labels_mask,
            batch_input_length,
        ) in train_dataloader:
            batch_PW_labels = batch_PW_labels.to(rank)
            batch_PPH_labels = batch_PPH_labels.to(rank)
            batch_IPH_labels = batch_IPH_labels.to(rank)
            batch_labels_mask = batch_labels_mask.to(rank)
            updater.zero_grad()
            prosody_word, prosody_phrase, intonation_phrase = ddp_model(
                batch_input_ids, rank
            )
            prosody_word_loss, prosody_phrase_loss, intonation_phrase_loss = loss_fn(
                prosody_word,
                batch_PW_labels,
                prosody_phrase,
                batch_PPH_labels,
                intonation_phrase,
                batch_IPH_labels,
                batch_labels_mask,
            )
            loss = prosody_word_loss + prosody_phrase_loss + intonation_phrase_loss
            loss.backward()
            updater.step()
            if rank == 0:
                PW_accuracy, PW_precision, PW_recall, PW_F1 = cal_metric(
                    batch_PW_labels, prosody_word, batch_input_length
                )
                PPH_accuracy, PPH_precision, PPH_recall, PPH_F1 = cal_metric(
                    batch_PPH_labels, prosody_phrase, batch_input_length
                )
                IPH_accuracy, IPH_precision, IPH_recall, IPH_F1 = cal_metric(
                    batch_IPH_labels, intonation_phrase, batch_input_length
                )
                log_dict = {
                    "PW loss": prosody_word_loss.item(),
                    "PPH loss": prosody_phrase_loss.item(),
                    "IPH loss": intonation_phrase_loss.item(),
                    "total loss": loss.item(),
                    "PW accuracy": PW_accuracy,
                    "PW precision": PW_precision,
                    "PW recall": PW_recall,
                    "PW F1": PW_F1,
                    "PPH accuracy": PPH_accuracy,
                    "PPH precision": PPH_precision,
                    "PPH recall": PPH_recall,
                    "PPH F1": PPH_F1,
                    "IPH accuracy": IPH_accuracy,
                    "IPH precision": IPH_precision,
                    "IPH recall": IPH_recall,
                    "IPH F1": IPH_F1,
                }
                print(
                    "training step: {}, total loss: {}, PW F1: {}, PPH F1: {}, IPH F1: {}".format(
                        train_step, loss.item(), PW_F1, PPH_F1, IPH_F1
                    )
                )
                summary.add_scalars("training", log_dict, train_step)
            train_step += 1
            dist.barrier()

        with torch.no_grad():
            ddp_model.eval()
            for (
                dev_batch_text,
                dev_batch_input_ids,
                dev_batch_PW_labels,
                dev_batch_PPH_labels,
                dev_batch_IPH_labels,
                dev_batch_labels_mask,
                dev_batch_input_length,
            ) in dev_dataloader:
                dev_batch_PW_labels = dev_batch_PW_labels.to(rank)
                dev_batch_PPH_labels = dev_batch_PPH_labels.to(rank)
                dev_batch_IPH_labels = dev_batch_IPH_labels.to(rank)
                dev_batch_labels_mask = dev_batch_labels_mask.to(rank)
                prosody_word, prosody_phrase, intonation_phrase = ddp_model(
                    dev_batch_input_ids, rank
                )
                (
                    prosody_word_loss,
                    prosody_phrase_loss,
                    intonation_phrase_loss,
                ) = loss_fn(
                    prosody_word,
                    dev_batch_PW_labels,
                    prosody_phrase,
                    dev_batch_PPH_labels,
                    intonation_phrase,
                    dev_batch_IPH_labels,
                    dev_batch_labels_mask,
                )
                loss = prosody_word_loss + prosody_phrase_loss + intonation_phrase_loss
                if rank == 0:
                    PW_accuracy, PW_precision, PW_recall, PW_F1 = cal_metric(
                        dev_batch_PW_labels, prosody_word, dev_batch_input_length
                    )
                    PPH_accuracy, PPH_precision, PPH_recall, PPH_F1 = cal_metric(
                        dev_batch_PPH_labels, prosody_phrase, dev_batch_input_length
                    )
                    IPH_accuracy, IPH_precision, IPH_recall, IPH_F1 = cal_metric(
                        dev_batch_IPH_labels, intonation_phrase, dev_batch_input_length
                    )

                    log_dict = {
                        "PW loss": prosody_word_loss.item(),
                        "PPH loss": prosody_phrase_loss.item(),
                        "IPH loss": intonation_phrase_loss.item(),
                        "total loss": loss.item(),
                        "PW accuracy": PW_accuracy,
                        "PW precision": PW_precision,
                        "PW recall": PW_recall,
                        "PW F1": PW_F1,
                        "PPH accuracy": PPH_accuracy,
                        "PPH precision": PPH_precision,
                        "PPH recall": PPH_recall,
                        "PPH F1": PPH_F1,
                        "IPH accuracy": IPH_accuracy,
                        "IPH precision": IPH_precision,
                        "IPH recall": IPH_recall,
                        "IPH F1": IPH_F1,
                    }
                    print(
                        "dev step: {}, total loss: {}, PW F1: {}, PPH F1: {}, IPH F1: {}".format(
                            dev_step, loss.item(), PW_F1, PPH_F1, IPH_F1
                        )
                    )
                    summary.add_scalars("dev", log_dict, dev_step)
                    dev_step += 1
                dist.barrier()
            if rank == 0:
                # only save epoch, multi-level encoder and multi-task learning decoder
                encoder = ddp_model.module.multi_level_encoder
                decoder = ddp_model.module.multi_task_learning_decoder
                torch.save(
                    [
                        i,
                        decoder.PW_decoder_hidden_dim,
                        decoder.PPH_decoder_hidden_dim,
                        decoder.IPH_decoder_hidden_dim,
                        encoder.state_dict(),
                        decoder.state_dict(),
                    ],
                    os.path.join(args.log_dir, "checkpoint_{}".format(i)),
                )
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
