"""
Main training entry point for NexoraLM.
Constructs datasets, initializes model and optimizer, runs trainer, and outputs artifacts.
"""

import os
import sys
import argparse
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import ModelConfig, TrainingConfig
from tokenizer.tokenizer import CharacterTokenizer
from data.preprocessing import load_text_file
from data.dataset import TextDataset, create_dataloader
from model.gpt_model import GPTLanguageModel
from training.trainer import Trainer, CosineWarmupScheduler
from training.checkpoint import CheckpointManager
from training.metrics import MetricsTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NexoraLM Language Model")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda')")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for metrics and plots")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test with minimal batches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Paths
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vocab_path = os.path.join(repo_root, "tokenizer", "vocabulary.json")
    train_path = os.path.join(repo_root, "data", "processed", "train.txt")
    val_path = os.path.join(repo_root, "data", "processed", "val.txt")

    # 2. Load Tokenizer & Vocabulary
    print("Loading tokenizer and vocabulary...")
    tokenizer = CharacterTokenizer()
    tokenizer.load_vocabulary(vocab_path)
    print(f"Loaded vocabulary with {tokenizer.vocab_size} tokens.")

    # 3. Model & Training Configurations
    if args.smoke_test:
        model_config = ModelConfig.from_tokenizer(
            tokenizer,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            context_length=64,
            dropout=0.1
        )
        training_config = TrainingConfig(
            batch_size=8,
            max_epochs=2,
            learning_rate=5e-4,
            eval_iters=5,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir
        )
    else:
        model_config = ModelConfig.from_tokenizer(
            tokenizer,
            embed_dim=256,
            num_layers=4,
            num_heads=4,
            context_length=256,
            dropout=0.1
        )
        training_config = TrainingConfig(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir
        )

    # Apply command-line overrides
    if args.epochs is not None:
        training_config.max_epochs = args.epochs
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    if args.lr is not None:
        training_config.learning_rate = args.lr
    if args.device is not None:
        training_config.device = args.device

    # Set random seed
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)

    # 4. Load Data & Prepare DataLoaders
    print("Loading preprocessed training and validation text...")
    train_text = load_text_file(train_path)
    val_text = load_text_file(val_path)

    if args.smoke_test:
        # Limit text slice for smoke testing
        train_text = train_text[:10000]
        val_text = val_text[:2000]

    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)

    train_dataset = TextDataset(train_tokens, context_length=model_config.context_length)
    val_dataset = TextDataset(val_tokens, context_length=model_config.context_length)
    print(f"Dataset ready: {len(train_dataset):,} train samples, {len(val_dataset):,} val samples.")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=training_config.num_workers
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=training_config.num_workers
    )

    # 5. Initialize Model
    print("Initializing NexoraLM GPTLanguageModel...")
    model = GPTLanguageModel(model_config)
    print(f"Total model parameters: {model.get_num_params():,}")

    # 6. Checkpoint Manager & Metrics Tracker
    checkpoint_mgr = CheckpointManager(training_config.checkpoint_dir)
    metrics_tracker = MetricsTracker(training_config.output_dir)

    # 7. Trainer setup
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        model_config=model_config,
        checkpoint_manager=checkpoint_mgr,
        metrics_tracker=metrics_tracker
    )

    # Resume training if requested
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        ckpt = checkpoint_mgr.load_checkpoint(
            checkpoint_path=args.resume,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            device=training_config.device
        )
        trainer.start_epoch = ckpt.get("epoch", 0) + 1
        trainer.global_step = ckpt.get("step", 0)
        trainer.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at epoch {trainer.start_epoch}, step {trainer.global_step}, best val loss: {trainer.best_val_loss:.4f}")

    # 8. Start training
    summary = trainer.train()
    print("Training finished successfully!")
    print(f"Best Validation Loss: {summary['best_val_loss']:.4f}")
    print(f"Metrics CSV: {summary['metrics_csv']}")
    print(f"Loss Plot: {summary['loss_plot']}")


if __name__ == "__main__":
    main()

