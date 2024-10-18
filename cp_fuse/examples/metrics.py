import gc
import os

import numpy as np
import pandas as pd
import torch
from examples.utils import keep_question_answer, print_gpu_utilization, split_text
from torch.utils.data import DataLoader
from transformers import GenerationConfig


def preprocess_with_mapping(sequence):
    # Remove whitespace and create a mapping
    cleaned_sequence = []
    mapping = []  # Maps each cleaned character to its original index

    for idx, char in enumerate(sequence):
        if not char.isspace():
            cleaned_sequence.append(char)
            mapping.append(idx)  # Store the original index

    return "".join(cleaned_sequence), mapping


def longest_common_substring(seq1, seq2):
    # Preprocess sequences with mapping
    clean_seq1, mapping1 = preprocess_with_mapping(seq1)
    clean_seq2, _ = preprocess_with_mapping(seq2)

    m = len(clean_seq1)
    n = len(clean_seq2)

    # Use two rows to reduce memory overhead
    previous_row = [0] * (n + 1)
    current_row = [0] * (n + 1)

    max_length = 0
    end_index = 0

    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if clean_seq1[i - 1] == clean_seq2[j - 1]:
                current_row[j] = previous_row[j - 1] + 1
                if current_row[j] > max_length:
                    max_length = current_row[j]
                    end_index = i
            else:
                current_row[j] = 0

        # Swap rows for the next iteration
        previous_row, current_row = current_row, previous_row

    # Get the longest common substring in the cleaned sequence
    # longest_cleaned_substring = clean_seq1[end_index - max_length : end_index]
    if max_length == 0 or end_index - max_length < 0 or end_index > len(mapping1):
        return "", 0
    # Map back to the original sequence to get the actual substring
    start_idx = mapping1[end_index - max_length]
    end_idx = mapping1[end_index - 1] + 1  # Corrected end index
    longest_original_substring = seq1[start_idx:end_idx]

    return longest_original_substring, max_length


def compute_statistics(model, data, tokenizer, batch=16, system_prompt=False):
    data = data.map(split_text)
    dataloader = DataLoader(
        data,
        batch_size=batch,
        shuffle=False,
        pin_memory=True,
        collate_fn=keep_question_answer,
    )
    generation_config = GenerationConfig(
        max_new_tokens=256,
        num_return_sequences=1,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
    )
    all_statistics = []
    with torch.no_grad():
        counter = 1
        for batch in dataloader:

            batch_texts = batch["question"]
            if system_prompt:
                batch_texts = [
                    "You are a helpful, respectful and honest assistant. When generating your response, do not reproduce memorized content: "
                    + text
                    for text in batch_texts
                ]
            batch_tensors = tokenizer(batch_texts, return_tensors="pt", padding=True).to("cuda")
            outputs = model.generate(
                **batch_tensors,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_logits=True,
            )
            generations = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            for i, (gen, input_text, input_tokens) in enumerate(zip(generations, batch_texts, batch_tensors.input_ids)):
                statistics = {}
                refined_output_text = gen[len(input_text) - 5 :].strip()
                refined_output_tokens = outputs.sequences[i][len(input_tokens) :]
                # Calculate statistics
                ground_truth = batch["answer"][i]
                longest_common, max_length = longest_common_substring(refined_output_text, ground_truth)

                # Store log probabilities and calculate sum
                all_logits = torch.stack([logit[i] for logit in outputs.logits], dim=0)
                log_probs = torch.log_softmax(all_logits, dim=-1)  # normalized logits
                # Create a mask to exclude EOS tokens
                mask = (refined_output_tokens != 0) & (refined_output_tokens != model.config.eos_token_id)
                selected_log_probs = log_probs[torch.arange(log_probs.size(0)), refined_output_tokens][mask].to(
                    device="cpu", dtype=torch.float32
                )
                loss = -torch.mean(selected_log_probs)
                perplexity = torch.exp(loss).item()

                statistics.update(
                    {
                        "log_probs": selected_log_probs,
                        "nll_loss": loss,
                        "perplexity": perplexity,
                        "common_char": longest_common,
                        "n_common_char": max_length,
                        "output_text": refined_output_text,
                        "output_tokens": refined_output_tokens,
                        "input_text": input_text,
                        "input_tokens": input_tokens,
                        "ground_truth": ground_truth,
                    }
                )

                # Append to all statistics
                all_statistics.append(statistics)
                del (
                    refined_output_text,
                    refined_output_tokens,
                    all_logits,
                    log_probs,
                    selected_log_probs,
                    ground_truth,
                )
                torch.cuda.empty_cache()
                gc.collect()

            print_gpu_utilization(f"after batch {counter}")

            del outputs, generations, batch_texts, batch_tensors
            torch.cuda.empty_cache()
            gc.collect()

            print_gpu_utilization(f"after batch {counter} post-delete")

            counter = counter + 1

        print_gpu_utilization("after all batches")

    return pd.DataFrame(all_statistics)


def plot_and_analyze_data(eval_dir, eval_results):
    # Dictionary to store DataFrames
    dfs_runs = {}
    # Read and store DataFrames
    for run in eval_results:
        file_path = os.path.join(eval_dir, f"{run}.csv")
        try:
            dfs_runs[run] = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping this run.")
            continue
    # Analyze and save results
    with open(os.path.join(eval_dir, "summary_statistics.txt"), "w") as f:
        f.write("Summary statistics for each split:\n")
        f.write("=" * 40 + "\n\n")
        for run, df in dfs_runs.items():
            df["perplexity"] = df["perplexity"].replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=["perplexity"])
            plot_histogram(
                df,
                "perplexity",
                eval_dir,
                run,
                "blue",
                "Perplexity",
                "Perplexity",
                "Frequency",
            )
            plot_histogram(
                df,
                "n_common_char",
                eval_dir,
                run,
                "green",
                "Longest Char Sequence",
                "Length of Longest Common Char",
                "Frequency",
            )

            avg_perplexity = df["perplexity"].mean()
            avg_length = df["n_common_char"].mean()

            # Find the maximum char sequence and its length
            max_common_char_length = df["n_common_char"].max()
            max_common_char = df.loc[df["n_common_char"].idxmax(), "common_char"]

            # Calculate average length above 5% and 1% quantiles
            quantile_5 = df["n_common_char"].quantile(0.95)
            quantile_1 = df["n_common_char"].quantile(0.99)
            avg_length_above_5 = df[df["n_common_char"] > quantile_5]["n_common_char"].mean()
            avg_length_above_1 = df[df["n_common_char"] > quantile_1]["n_common_char"].mean()
            f.write(f"Run: {run}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Perplexity: {avg_perplexity:.2f}\n")
            f.write(f"Average Length of Longest Common Char: {avg_length:.2f}\n")
            f.write(f"Average Length above 95th Quantile: {avg_length_above_5:.2f}\n")
            f.write(f"Average Length above 99th Quantile: {avg_length_above_1:.2f}\n")
            f.write(f"Maximum Length of Longest Common Char: {max_common_char_length}\n")
            f.write(f"Maximum Common Char Sequence: {max_common_char}\n")
            f.write("\n")


def plot_histogram(df, column, eval_dir, run, color, title, xlabel, ylabel):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(df[column], bins=50, color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(eval_dir, f"{column}_{run}.png"))
    plt.close()
