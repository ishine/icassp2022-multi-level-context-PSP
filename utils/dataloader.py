import torch
from torch.utils.data import Dataset
import random


def get_context_window_sample_index(idx, number_total_samples, window_size):
    if window_size == 1:
        # we only need one sample
        return [idx]
    if number_total_samples < window_size:
        # number of total samples is less than the window size, return all samples
        return range(window_size)
    if idx < window_size // 2:
        # there isn't enough samples before the selected sample
        return range(window_size)
    if idx + window_size // 2 > number_total_samples:
        # there isn't enough sample after the selected sample
        return range(number_total_samples - window_size, number_total_samples)
    return range(idx - window_size // 2, idx - window_size // 2 + window_size)


def get_single_item(samples, idx, max_length):
    raw_text = samples[idx]
    tokens = []
    PW_label = [0] * max_length
    PPH_label = [0] * max_length
    IPH_label = [0] * max_length
    label_mask = [0] * max_length

    i = 0
    j = 0
    while i < len(raw_text):
        if raw_text[i] == "#" and i + 1 < len(raw_text):
            label = int(raw_text[i + 1])
            if label == 1:
                # we get a PW label
                PW_label[j - 1] = 1
            elif label == 2:
                # we get a PPH label
                # it should be added to both PW label and PPH label
                PW_label[j - 1] = 1
                PPH_label[j - 1] = 1
            elif label == 3:
                # we get a IPH label
                # it should be added to both PW label, PPH label and IPH label
                PW_label[j - 1] = 1
                PPH_label[j - 1] = 1
                IPH_label[j - 1] = 1
            else:
                raise ValueError("Wrong prosody label!")
            i += 2
        else:
            tokens.append(raw_text[i])
            label_mask[j] = 1
            i += 1
            j += 1
    text = "".join(tokens)
    if len(tokens) > max_length:
        sample_length = max_length
    else:
        sample_length = len(tokens)
    return (
        text,
        PW_label,
        PPH_label,
        IPH_label,
        label_mask,
        sample_length,
    )


class ProsodyStructurePredictionDataset(Dataset):
    def __init__(self, dataset_path, context_window_size, tokenizer, max_length):
        super(ProsodyStructurePredictionDataset, self).__init__()
        self.dataset_path = dataset_path
        self.context_window_size = context_window_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = None

    def __load_dataset(self):
        if self.samples is None:
            with open(self.dataset_path) as f:
                self.samples = [line.strip() for line in f.readlines()]

    def __getitem__(self, idx):
        # return list of texts, tokenized texts and labels for a single sentence
        self.__load_dataset()
        texts = []
        PW_labels = []
        PPH_labels = []
        IPH_labels = []
        labels_mask = []
        input_length = []
        for (text, PW_label, PPH_label, IPH_label, label_mask, sample_length) in map(
            lambda x: get_single_item(self.samples, x, self.max_length),
            get_context_window_sample_index(idx, len(self), self.context_window_size),
        ):
            texts.append(text)
            PW_labels.append(PW_label)
            PPH_labels.append(PPH_label)
            IPH_labels.append(IPH_label)
            labels_mask.append(label_mask)
            input_length.append(sample_length)
        return (
            texts,
            self.tokenizer(
                texts,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
                add_special_tokens=False,
            ),
            PW_labels,
            PPH_labels,
            IPH_labels,
            labels_mask,
            input_length,
        )

    def __len__(self):
        self.__load_dataset()
        return len(self.samples)


class ProsodyStructurePredictionInferenceDataset(Dataset):
    def __init__(self, input_path, context_window_size, tokenizer, max_length):
        super(ProsodyStructurePredictionInferenceDataset, self).__init__()
        self.input_path = input_path
        self.context_window_size = context_window_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(input_path) as f:
            self.samples = [line.strip() for line in f.readlines()]
        self.indexs = [x for x in range(0, len(self.samples), context_window_size)]
        self.collate_fn = DatasetCollateFn()

    def __getitem__(self, idx):
        # in inference, we don't need to sample the samples like training
        texts = []
        PW_labels = []
        PPH_labels = []
        IPH_labels = []
        labels_mask = []
        input_length = []
        for i in range(
            self.indexs[idx],
            min(len(self.samples), self.indexs[idx] + self.context_window_size),
        ):
            (
                text,
                PW_label,
                PPH_label,
                IPH_label,
                label_mask,
                sample_length,
            ) = get_single_item(self.samples, i, self.max_length)
            texts.append(text)
            PW_labels.append(PW_label)
            PPH_labels.append(PPH_label)
            IPH_labels.append(IPH_label)
            labels_mask.append(label_mask)
            input_length.append(sample_length)
        return self.collate_fn(
            [
                (
                    texts,
                    self.tokenizer(
                        texts,
                        max_length=self.max_length,
                        padding="max_length",
                        return_tensors="pt",
                        truncation=True,
                        add_special_tokens=False,
                    ),
                    PW_labels,
                    PPH_labels,
                    IPH_labels,
                    labels_mask,
                    input_length,
                )
            ]
        )

    def __len__(self):
        return len(self.samples) - self.context_window_size + 1


class DatasetCollateFn:
    def __call__(self, data):
        batch_text = []
        batch_PW_labels = []
        batch_PPH_labels = []
        batch_IPH_labels = []
        batch_input_ids = []
        batch_labels_mask = []
        batch_input_length = []
        for (
            texts,
            input_ids,
            PW_label,
            PPH_label,
            IPH_label,
            labels_mask,
            input_length,
        ) in data:
            batch_text.append(texts)
            batch_input_ids.append(input_ids)
            batch_PW_labels.append(PW_label)
            batch_PPH_labels.append(PPH_label)
            batch_IPH_labels.append(IPH_label)
            batch_labels_mask.append(labels_mask)
            batch_input_length.append(input_length)
        return (
            batch_text,
            batch_input_ids,
            torch.LongTensor(batch_PW_labels),
            torch.LongTensor(batch_PPH_labels),
            torch.LongTensor(batch_IPH_labels),
            torch.IntTensor(batch_labels_mask),
            torch.IntTensor(batch_input_length),
        )
