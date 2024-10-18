import os
import random
import re

import torch
from datasets import load_dataset, load_from_disk
from examples.config import (
    DATA_DIR_CACHE,
    DATA_DIR_EVAL,
    DATA_DIR_MODELS,
    DATA_DIR_STORAGE,
)
from torch.utils.data import IterableDataset


# Utility function to ensure required directories exist
def ensure_directories():
    """
    Ensures that all necessary directories exist, creating them if they do not.
    """
    directories = [DATA_DIR_STORAGE, DATA_DIR_MODELS, DATA_DIR_EVAL, DATA_DIR_CACHE]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' checked/created.")


# Base class for data processors
class DataProcessor:
    def __init__(self, seed=42, sep_token="[SEP]"):
        self.special_token = sep_token
        self.seed = seed

    def filter_and_shuffle(self, dataset, filter_fn=None, limit=None):
        """
        Filters, shuffles, and optionally limits the dataset.
        """
        if filter_fn:
            dataset = dataset.filter(filter_fn)
        dataset = dataset.shuffle(seed=self.seed)
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        return dataset


# Data processor for instructional code dataset
class CodeInstructionsDataProcessor(DataProcessor):
    def load_and_process_data(self):
        dataset = load_dataset(
            "Nan-Do/instructional_code-search-net-python",
            download_mode="force_redownload",
        )["train"]
        dataset = dataset.filter(lambda entry: entry["RESPONSE"].startswith("def "))
        dataset = dataset.map(lambda entry: {"RESPONSE": self.remove_docstring(entry["RESPONSE"])})
        dataset = dataset.train_test_split(test_size=0.1, seed=self.seed)

        train_dataset = self.filter_and_shuffle(dataset["train"], limit=100000)
        test_dataset = self.filter_and_shuffle(dataset["test"], limit=10000)

        train_dataset = train_dataset.map(lambda x: {"text": x["INSTRUCTION"] + self.special_token + x["RESPONSE"]})
        test_dataset = test_dataset.map(lambda x: {"text": x["INSTRUCTION"] + self.special_token + x["RESPONSE"]})

        return train_dataset.select_columns(["text"]), test_dataset.select_columns(["text"])

    def remove_docstring(self, code):
        """
        Remove the docstring from the given Python code.
        """
        # Regular expression to match the docstring
        docstring_pattern = re.compile(r'""".*?"""|\'\'\'.*?\'\'\'', re.DOTALL)
        code = re.sub(docstring_pattern, "", code)
        code = re.sub(r"\n\s*\n", "\n", code)
        return code


# Data processor for math abstracts dataset
class MathAbstractsDataProcessor(DataProcessor):
    def load_and_process_data(self):
        dataset = load_dataset("math-ai/AutoMathText", "arxiv-0.50-to-1.00")["train"]
        dataset = dataset.select_columns(["title", "abstract"]).filter(lambda x: len(x["abstract"]) > 600)
        dataset = dataset.train_test_split(test_size=0.1, seed=self.seed)

        train_dataset = self.filter_and_shuffle(dataset["train"], limit=100000)
        test_dataset = self.filter_and_shuffle(dataset["test"], limit=10000)

        train_dataset = train_dataset.map(lambda x: {"text": x["title"] + self.special_token + x["abstract"]})
        test_dataset = test_dataset.map(lambda x: {"text": x["title"] + self.special_token + x["abstract"]})

        return train_dataset.select_columns(["text"]), test_dataset.select_columns(["text"])


# Data processor for storytelling dataset
class StoryTellingDataProcessor(DataProcessor):
    def load_and_process_data(self):
        dataset = load_dataset("euclaise/writingprompts")
        train_dataset = self.filter_and_shuffle(
            dataset["train"],
            filter_fn=lambda x: len(x["prompt"]) > 70 and len(x["story"]) < 2000,
        )
        test_dataset = self.filter_and_shuffle(
            dataset["test"],
            filter_fn=lambda x: len(x["prompt"]) > 70 and len(x["story"]) < 2000,
        )

        train_dataset = train_dataset.map(lambda x: {"text": x["prompt"] + self.special_token + x["story"]})
        test_dataset = test_dataset.map(lambda x: {"text": x["prompt"] + self.special_token + x["story"]})

        return train_dataset.select_columns(["text"]), test_dataset.select_columns(["text"])


# Dictionary to map dataset names to their corresponding processor classes
DATA_PROCESSORS = {
    "CodeInstructions": CodeInstructionsDataProcessor,
    "MathAbstracts": MathAbstractsDataProcessor,
    "StoryTelling": StoryTellingDataProcessor,
}


# Class for loading datasets with optional subsampling and splitting
class DatasetLoader:
    def __init__(self, seed=42):
        self.seed = seed

    def load_or_create_datasets(self, dataset_name, ntrain=0, k=2):
        """
        Loads or creates and splits the dataset specified by `dataset_name`.
        """
        processor_class = DATA_PROCESSORS.get(dataset_name)
        if not processor_class:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        processor = processor_class(seed=self.seed)
        train_full_path, test_full_path = [
            os.path.join(DATA_DIR_STORAGE, f"{split}_{dataset_name}") for split in ["train_full", "test_full"]
        ]

        # Load datasets from disk or create them if not available
        try:
            train_full, test_full = load_from_disk(train_full_path), load_from_disk(test_full_path)
            print("Datasets loaded from disk.")
        except FileNotFoundError:
            print("Creating datasets from scratch.")
            train_full, test_full = processor.load_and_process_data()
            train_full.save_to_disk(train_full_path)
            test_full.save_to_disk(test_full_path)

        # Optionally subsample and split training data
        train_full = train_full.select(range(min(ntrain, len(train_full)))) if ntrain else train_full
        test_full = test_full.select(range(min(ntrain, len(test_full)))) if ntrain else test_full

        # Split into multiple datasets based on `k`
        train_datasets = [train_full.shard(num_shards=k, index=i) for i in range(k)]
        return train_datasets, train_full, test_full


# Iterable dataset with constant-length chunks
class ConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        concat_token_id=None,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = concat_token_id if concat_token_id is not None else tokenizer.eos_token_id
        print(f"Concat token ID (EOS token): {self.concat_token_id}")
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)["text"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break

            # Tokenize inputs and generate sequences
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            examples = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for input_ids in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids)),
                }

    def get_tokenizer(self):
        return self.tokenizer
