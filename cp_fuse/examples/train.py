import argparse

from examples.config import DATA_DIR_CACHE
from examples.data import ConstantLengthDataset, DatasetLoader, ensure_directories
from examples.utils import chars_token_ratio, get_total_tokens
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def parse_arguments():
    """
    Parses command-line arguments for model training configuration.
    """
    parser = argparse.ArgumentParser(description="Fine-tuning script for language model")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model checkpoint path",
    )
    parser.add_argument("--dataset_name", type=str, default="MathAbstracts", help="Name of the dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="Sequence length for the model input",
    )
    parser.add_argument("--n_train_samples", type=int, default=9000, help="Number of training samples")
    parser.add_argument(
        "--num_splits",
        type=int,
        default=2,
        help="Number of dataset splits for training",
    )
    parser.add_argument("--split", type=int, default=1, help="Specify the training split")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per training step")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for training")
    parser.add_argument("--quantize", action="store_true", help="Enable 8-bit quantization (LoRA)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Steps to accumulate gradients",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="Save model every specified number of epochs",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1,
        help="Evaluate model every specified number of epochs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save model checkpoints",
    )
    return parser.parse_args()


def load_datasets(args):
    """
    Loads or creates the datasets required for training and validation.
    """
    dataloader = DatasetLoader()
    train_dataset, train_full, validation = dataloader.load_or_create_datasets(
        dataset_name=args.dataset_name, ntrain=args.n_train_samples, k=args.num_splits
    )
    datasets = {"train": train_full, "validation": validation}
    if args.split > 0:
        datasets["train"] = train_dataset[args.split - 1]
    print(f"Train set size: {len(datasets['train'])}. Validation set size: {len(datasets['validation'])}")
    return datasets


def init_tokenizer(model_checkpoint):
    """
    Initializes the tokenizer with special tokens added.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        torch_dtype="auto",
        cache_dir=DATA_DIR_CACHE,
        use_cache=True,
        trust_remote_code=True,
    )
    special_tokens_dict = {"sep_token": "[SEP]", "pad_token": "[PAD]"}
    tokenizer.add_special_tokens(special_tokens_dict)
    print("Special tokens:", tokenizer.special_tokens_map)
    return tokenizer


def prepare_datasets(datasets, tokenizer, args):
    """
    Prepares and configures training and validation datasets.
    """
    chars_per_token = chars_token_ratio(datasets["train"], tokenizer)
    print(f"Character to token ratio: {chars_per_token:.2f}")
    train_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=datasets["train"],
        infinite=True,
        chars_per_token=chars_per_token,
        seq_length=args.seq_length,
        num_of_sequences=512,
    )
    validation_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=datasets["validation"],
        infinite=False,
        chars_per_token=chars_per_token,
        seq_length=args.seq_length,
        num_of_sequences=512,
    )
    total_tokens = get_total_tokens(datasets["train"], tokenizer)
    training_examples = total_tokens // args.seq_length
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    max_steps = max(1, int(training_examples / effective_batch_size * args.epochs))
    print(f"Total tokens: {total_tokens}, Training examples: {training_examples}, Max steps: {max_steps}")
    return train_dataset, validation_dataset, max_steps


def init_model(args, tokenizer):
    """
    Initializes the language model with optional quantization (LoRA).
    """
    if args.quantize:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_checkpoint,
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True,
        )
        peft_config = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM")
        model.add_adapter(peft_config, adapter_name="adapter_1")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    return model


def create_training_args(args, output_dir, max_steps):
    """
    Configures and returns training arguments.
    """
    # Calculate steps per epoch, ensuring no division by zero
    steps_per_epoch = max(1, max_steps // args.epochs)
    eval_steps = max(1, int(steps_per_epoch * args.eval_freq))
    save_steps = max(1, int(steps_per_epoch * args.save_freq))
    logging_steps = max(1, steps_per_epoch // 10)

    # Return configured TrainingArguments
    return TrainingArguments(
        output_dir=output_dir,
        dataloader_drop_last=True,
        overwrite_output_dir=True,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_total_limit=1,
        fp16=args.quantize if hasattr(args, "quantize") else False,
        bf16=False,
        optim="adamw_bnb_8bit",
        seed=args.seed,
        warmup_steps=50,
    )


def main():
    args = parse_arguments()
    ensure_directories()
    datasets = load_datasets(args)
    tokenizer = init_tokenizer(args.model_checkpoint)
    train_dataset, validation_dataset, max_steps = prepare_datasets(datasets, tokenizer, args)
    model = init_model(args, tokenizer)
    output_dir = f"{args.output_dir}/{args.dataset_name}_split_{args.split}"
    training_args = create_training_args(args, output_dir, max_steps)
    trainer = Trainer(
        model=model.half(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    # Save the final model after training completes
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Final model saved to {output_dir}")


if __name__ == "__main__":
    main()
