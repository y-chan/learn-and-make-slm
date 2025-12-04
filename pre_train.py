from argparse import ArgumentParser
import os
import itertools
from pathlib import Path

import datasets
import tiktoken
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import SLMConfig
from dataset import SimpleStoriesBothDataset, dataset_collate, random_end_lengths
from models.transformer.decoder import GPT2Decoder
from utils.checkpoint import latest_checkpoint_path, load_checkpoint, save_checkpoint
from utils.tools import to_device


def validate(
    model: GPT2Decoder,
    test_loader: torch.utils.data.DataLoader,
    test_writer: SummaryWriter,
    tokenizer: tiktoken.Encoding,
    epoch: int | None = None,
    step: int | None = None,
):
    assert (epoch is not None and step is None) or (epoch is None and step is not None), (
        "Either epoch or step must be provided, but not both"
    )

    model.eval()

    all_loss = 0

    with torch.no_grad():
        for _batch in tqdm(test_loader, desc="Validation", dynamic_ncols=True, position=2):
            batch = to_device(_batch, next(model.parameters()).device)
            # randomized_lengths = random_end_lengths(batch["lengths"] - 1)
            seq_lengths = batch["lengths"] - 1
            output = model(batch["tokens_ids"][:, :-1], seq_lengths)
            loss = model.loss(output, batch["tokens_ids"][:, 1:], seq_lengths)
            all_loss += loss.item()

            if epoch is not None:
                for i in range(min(batch["tokens_ids"].size(0), 5)):
                    tokens_ids = batch["tokens_ids"][i:i+1]
                    output_tokens_ids = model.infer(starts=tokens_ids[:, :5], max_token_count=200)
                    gt_text = tokenizer.decode(tokens_ids)
                    output_text = tokenizer.decode(tokens_ids[:5].tolist() + output_tokens_ids.tolist())
                    test_writer.add_text("Text/GT", gt_text, 0)
                    test_writer.add_text("Text/Test_Output", output_text, epoch)

    test_writer.add_scalar(
        f"Loss/{'Epoch' if epoch is not None else 'Step'}_Test",
        all_loss / len(test_loader),
        epoch if epoch is not None else step,
    )


def train(
    config: SLMConfig,
    model: GPT2Decoder,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    tokenizer: tiktoken.Encoding,
    start_epoch: int,
):
    # Initialize SummaryWriters
    train_log_dir = config.path.log_dir
    test_log_dir = config.path.log_dir / "test"
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(test_log_dir, exist_ok=True)

    train_writer = SummaryWriter(log_dir=train_log_dir)
    test_writer = SummaryWriter(log_dir=test_log_dir)

    total_bar = tqdm(
        total=config.train.epochs * len(train_loader),
        position=0,
        desc="Steps",
        dynamic_ncols=True,
    )
    total_bar.n = (start_epoch - 1) * len(train_loader)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_decay, last_epoch=start_epoch - 2)

    interrupted = False

    for epoch in range(start_epoch, config.train.epochs + 1):
        epoch_loss = 0
        for _batch in tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True, position=1):
            try:
                batch = to_device(_batch, next(model.parameters()).device)
                randomized_lengths = random_end_lengths(batch["lengths"] - 1)
                output = model(batch["tokens_ids"][:, :-1], randomized_lengths)
                loss = model.loss(output, batch["tokens_ids"][:, 1:], randomized_lengths)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

                if total_bar.n % config.train.logging_steps == 0:
                    tqdm.write(f"Step {total_bar.n}, Loss: {loss.item()}")
                    train_writer.add_scalar("Loss/Step_Train", loss.item(), total_bar.n)
                    validate(model, test_loader, test_writer, tokenizer, step=total_bar.n)

                total_bar.update()
            except KeyboardInterrupt:
                if interrupted:
                    raise KeyboardInterrupt
                tqdm.write("KeyboardInterrupt detected. Please wait for the checkpoint to be saved...")
                tqdm.write("If you want to quit immediately, please press Ctrl+C again. But the checkpoint will not be saved.")
                interrupted = True

        train_writer.add_scalar("Loss/Epoch_Train", epoch_loss / len(train_loader), epoch)
        train_writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        if total_bar.n % config.train.validation_epochs == 0:
            validate(model, test_loader, test_writer, tokenizer, epoch=epoch)

        if epoch % config.train.save_epochs == 0 or interrupted:
            save_checkpoint(model, optimizer, epoch, os.path.join(config.path.log_dir, f"checkpoint_{epoch}.pth"))

        if interrupted:
            break

    # Close SummaryWriters
    train_writer.close()
    test_writer.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, default="config/simple_stories.yaml")
    args = parser.parse_args()
    config = SLMConfig.load(args.config)

    # Set random seed
    torch.manual_seed(config.train.manual_seed)
    torch.cuda.manual_seed(config.train.manual_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = tiktoken.get_encoding(config.tokenizer)

    model = GPT2Decoder(
        tokenizer.n_vocab,
        config.model.n_layers,
        config.model.d_model,
        config.model.n_heads,
        tokenizer.eot_token
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), betas=config.train.betas, lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )

    model_dir = Path(config.path.log_dir)
    try:
        start_epoch = (
            load_checkpoint(
                latest_checkpoint_path(model_dir, "checkpoint_*.pth"),
                model,
                optimizer,
            )
            + 1
        )
    except Exception as e:
        print(e)
        start_epoch = 1

    if config.dataset == "SimpleStories-Both":
        def encode_story(xs):
            xs.update({"story": [tokenizer.encode(x + "<|endoftext|>", allowed_special={"<|endoftext|>"}) for x in xs["story"]]})
            return xs

        train_dataset = SimpleStoriesBothDataset(subset="train")
        train_dataset = train_dataset.map(encode_story, batched=True)

        test_dataset = SimpleStoriesBothDataset(subset="test")
        test_dataset = test_dataset.map(encode_story, batched=True)
    elif "SimpleStories" in config.dataset:
        train_dataset = datasets.load_dataset(config.dataset, split="train")
        train_dataset = train_dataset.map(encode_story, batched=True)
        test_dataset = datasets.load_dataset(config.dataset, split="test")
        test_dataset = test_dataset.map(encode_story, batched=True)
    else:
        raise ValueError(f"Dataset {config.dataset} not supported")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=lambda x: dataset_collate(x, torch_convert=True),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=lambda x: dataset_collate(x, torch_convert=True),
    )

    train(config, model, optimizer, train_loader, test_loader, tokenizer, start_epoch)


if __name__ == "__main__":
    main()
