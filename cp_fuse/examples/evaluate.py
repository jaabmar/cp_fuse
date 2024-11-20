import argparse
import gc
import os
import time

import torch
from cp_fuse.cp_model import CPModel
from examples.config import DATA_DIR_CACHE
from examples.data import DatasetLoader
from examples.metrics import compute_statistics, plot_and_analyze_data
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    """
    Parses command-line arguments for model evaluation configuration.
    """
    parser = argparse.ArgumentParser(description="Evaluation script for CP_Fuse models")
    parser.add_argument("--model_checkpoint1", type=str, required=True, help="Path to the first model checkpoint")
    parser.add_argument("--model_checkpoint2", type=str, help="Path to the second model checkpoint for CPModel")
    parser.add_argument("--dataset_name", type=str, required=True, default="MathAbstracts", help="Name of the dataset")
    parser.add_argument("--n_test_samples", type=int, default=500, help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="./eval", help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size for grid search in CPModel")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output during evaluation")
    return parser.parse_args()


def init_tokenizer(model_checkpoint):
    """
    Initializes the tokenizer with special tokens added.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        torch_dtype="auto",
        cache_dir=DATA_DIR_CACHE,
        trust_remote_code=True,
    )
    special_tokens_dict = {"sep_token": "[SEP]", "pad_token": "[PAD]"}
    tokenizer.add_special_tokens(special_tokens_dict)
    print("Special tokens:", tokenizer.special_tokens_map)
    return tokenizer


def load_models(args, tokenizer):
    """
    Loads the primary model or CPModel based on provided arguments.
    """
    if args.model_checkpoint2:
        print(f"Loading CPModel with {args.model_checkpoint1} and {args.model_checkpoint2}")
        model1 = AutoModelForCausalLM.from_pretrained(
            args.model_checkpoint1, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        ).half()
        model1.resize_token_embeddings(len(tokenizer))
        model2 = AutoModelForCausalLM.from_pretrained(
            args.model_checkpoint2, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        ).half()
        model2.resize_token_embeddings(len(tokenizer))

        model_name = f"{args.dataset_name}_cp_model"
        return (
            CPModel(
                model1=model1,
                model2=model2,
                grid_size=args.grid_size,
                verbose=args.verbose,
            ),
            model_name,
        )
    else:
        print(f"Loading model from {args.model_checkpoint1}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_checkpoint1, device_map="auto", trust_remote_code=True
        ).half()
        model.resize_token_embeddings(len(tokenizer))
        model_name = f"{args.dataset_name}_single_model"
        return model, model_name


def evaluate_datasets(train_dataset, validation, model, tokenizer, eval_dir, batch_size):
    """
    Evaluates multiple datasets and saves results to CSV files.
    """
    eval_folder_names = []
    datasets = {"train": train_dataset, "validation": validation}

    for name, data in datasets.items():
        file_name = os.path.join(eval_dir, f"{name}.csv")
        if not os.path.isfile(file_name):
            print(f"Evaluating {name} set...")
            start_time = time.time()
            eval_res = compute_statistics(model=model, data=data, tokenizer=tokenizer, batch=batch_size)
            eval_res.to_csv(file_name)
            print(f"Evaluation of {name} completed in {time.time() - start_time:.2f} seconds.")
            del eval_res
            torch.cuda.empty_cache()
            gc.collect()
        eval_folder_names.append(f"{name}")

    plot_and_analyze_data(eval_dir, eval_folder_names)


def main():
    args = parse_arguments()

    # Set up evaluation directory
    eval_dir = os.path.join(args.output_dir, f"{args.dataset_name}_evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Initialize tokenizer
    tokenizer = init_tokenizer(args.model_checkpoint1)

    # Load datasets
    dataloader = DatasetLoader()
    _, train_dataset, validation_dataset = dataloader.load_or_create_datasets(
        dataset_name=args.dataset_name,
        ntrain=args.n_test_samples,
    )

    # Load model(s)
    model, _ = load_models(args, tokenizer)

    # Run evaluation
    evaluate_datasets(
        train_dataset=train_dataset,
        validation=validation_dataset,
        model=model,
        tokenizer=tokenizer,
        eval_dir=eval_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
