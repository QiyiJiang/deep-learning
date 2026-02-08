import json
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

def _build_loss_mask(input_ids, tokenizer):
    bos_id = tokenizer(
        f"{tokenizer.bos_token}assistant",
        add_special_tokens=False,
    ).input_ids
    eos_id = tokenizer(
        tokenizer.eos_token,
        add_special_tokens=False,
    ).input_ids

    ids = input_ids.tolist() if input_ids.dim() > 0 else [input_ids.item()]
    loss_mask = [0] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i : i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            while end < len(ids):
                if ids[end : end + len(eos_id)] == eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(eos_id), len(ids))):
                loss_mask[j] = 1
            i = end + len(eos_id) if end < len(ids) else len(ids)
        
        else:
            i += 1

    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        for i in range(len(ids)):
            if ids[i] == pad_id:
                loss_mask[i] = 0

    return torch.tensor(loss_mask, dtype=torch.long, device=input_ids.device)

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
    """流式预训练数据集。num_workers>0 时自动按行分片；建议传 tokenizer_path 避免多进程 tokenizer 问题。"""

    def __init__(self, file_path, tokenizer=None, tokenizer_path=None, max_length=512):
        assert tokenizer is not None or tokenizer_path is not None, "传 tokenizer 或 tokenizer_path 之一"
        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.file_path = file_path

    def _get_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.tokenizer_path)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        tokenizer = self._get_tokenizer()

        with open(self.file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx % num_workers != worker_id:
                    continue
                try:
                    data = json.loads(line)
                    text = data["text"]
                except (json.JSONDecodeError, KeyError):
                    continue
                enc = tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].squeeze(0)
                labels = input_ids.clone()
                loss_mask = (input_ids != tokenizer.pad_token_id).long()
                yield input_ids, labels, loss_mask


class SimpleSFTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data["conversations"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        text = self.tokenizer.apply_chat_template(
            text,
            tokenize=False,
            add_generation_prompt=False
        )
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" 
        )
        input_ids = enc["input_ids"].squeeze(0)
        loss_mask = _build_loss_mask(input_ids, self.tokenizer)
        labels = input_ids.clone()
        
        return input_ids, labels, loss_mask 




class SFTDataset(IterableDataset):
    """流式 SFT 数据集。num_workers>0 时自动按行分片；建议传 tokenizer_path 避免多进程 tokenizer 问题。"""

    def __init__(self, file_path, tokenizer=None, tokenizer_path=None, max_length=512):
        assert tokenizer is not None or tokenizer_path is not None, "传 tokenizer 或 tokenizer_path 之一"
        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.file_path = file_path

    def _get_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.tokenizer_path)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        tokenizer = self._get_tokenizer()

        with open(self.file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx % num_workers != worker_id:
                    continue
                try:
                    data = json.loads(line)
                    conv = data["conversations"]
                except (json.JSONDecodeError, KeyError):
                    continue
                try:
                    text = tokenizer.apply_chat_template(
                        conv,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception:
                    continue
                enc = tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].squeeze(0)
                loss_mask = _build_loss_mask(input_ids, tokenizer)
                labels = input_ids.clone()
                yield input_ids, labels, loss_mask
