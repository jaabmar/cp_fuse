import torch
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit


def print_gpu_memory_stats(prefix=""):
    # Check if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all kernels to finish
        allocated = torch.cuda.memory_allocated()  # Current memory allocated in bytes
        cached = (
            torch.cuda.memory_reserved()
        )  # Current memory cached (reserved) in bytes
        print(f"{prefix}Memory Allocated: {allocated / 1e9:.2f} GB")
        print(f"{prefix}Memory Cached: {cached / 1e9:.2f} GB")


def print_gpu_utilization(prefix="now"):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied {prefix}: {info.used // 1024**2} MB.")


def get_total_tokens(dataset, tokenizer):
    """
    Estimate the total number of tokens in the dataset.
    """
    nb_examples = len(dataset)
    total_tokens = 0
    for _, example in zip(range(nb_examples), iter(dataset)):
        text = example["text"]
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_tokens


def chars_token_ratio(
    dataset,
    tokenizer,
    nb_examples=400,
):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in zip(range(nb_examples), iter(dataset)):
        text = example["text"]
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def split_text(example):
    # Splitting the text on '[SEP]'
    split_parts = example["text"].split("[SEP]", 1)  # Limit to one split

    if len(split_parts) != 2:
        raise ValueError("Expected exactly one '[SEP]' in each text entry.")

    return {
        "question": f"{split_parts[0]}[SEP]",
        "answer": split_parts[1].strip(),
    }


def keep_question_answer(batch):
    # This keeps only the 'question' and 'answer' fields from each item in the batch
    return {
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
    }
