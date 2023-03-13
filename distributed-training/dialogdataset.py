# 引入相应的包 Importing libraries
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os, time
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration


# rich: for a better display on terminal

class DialogDataSet(Dataset):
    """
    创建一个自定义的数据集，用于训练，必须包括两个字段：输入(如source_text)、输出（如target_text）
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
            self, data_list, tokenizer: T5Tokenizer, max_length, div_size=1
    ):
        """
        Initializes a Dataset class

        Args:
            data_list: 输入的数据列表, dict item
            tokenizer (transformers.tokenizer): Transformers tokenizer

        """
        length = len(data_list)
        self.data_list = data_list[:length - length % div_size]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.data_list)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        data_item_dict = self.data_list[index]

        source_text = data_item_dict["source"]
        target_text = data_item_dict["target"]

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
            "source_text": source_text
        }


print("end...")
