# Multi-level Contextual Information Prosody Structure Prediction
Code for paper: Improving Mandarin Prosody Structure Prediction with Multi-level Contextual Information

## Requirements
```
pytorch, sklearn, transformers
```

## Usage

### Training

#### Dataset preparation
We use the same prosodic labeling method as [Pan et al.,2019](https://www.isca-speech.org/archive_v0/Interspeech_2019/pdfs/1400.pdf). Example training samples are `dataset/train.txt` and `dataset/dev.txt`.

Prosody structure labels should be inserted into text. 
That is if there's a prosody word boundary after a Chinese character, '#1' should be inserted after the character. 
If there's a prosody phrase boundary, '#2' should be inserted. 
For a intonation phrase boundary, it's '#3'.
Each line in the file containing samples is considered one sentence.
Since surrounding samples of one training sample is considered to be the context, the training samples should not be shuffled, or the model will get wrong contextual information from surrounding sentences. 
For example, the $t+1$-th line in the training samples is the next sentence of $t$-th line.

#### Start Training
```python3
python3 train.py --train_dataset [PATH TO TRAINING DATASET] --dev_dataset [PATH TO DEV DATASET]
```
For more information about parameters of `train.py`, see `train_args.py`.

### Inference
#### Dataset preparation
Dataset for inference is the same as training dataset. Of course there is no prosody structure labels.
#### Start inference
```python
python3 inference.py --model [PATH TO CHECKPOINT] --input [PATH TO DATASET FOR INFERENCE] --device [GPU DEVICE FOR INFERENCE]
```
