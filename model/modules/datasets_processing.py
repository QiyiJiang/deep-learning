from cProfile import label
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset

class SimplePretrainDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data["text"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" 
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        loss_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, labels, loss_mask 


class PretrainDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data["text"]
                enc = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt" 
                )
                input_ids = enc["input_ids"].squeeze(0)
                labels = input_ids
                loss_mask = (input_ids != self.tokenizer.pad_token_id).long()
                yield input_ids, labels, loss_mask

