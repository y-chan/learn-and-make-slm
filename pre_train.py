from argparse import ArgumentParser
import os
from pathlib import Path

import datasets
import tiktoken
import torch
from tqdm import tqdm

from config import SLMConfig
from dataset import dataset_collate, random_end_lengths
from models.transformer.decoder import Decoder
from utils.checkpoint import latest_checkpoint_path, load_checkpoint, save_checkpoint
from utils.tools import to_device

def train(
    config: SLMConfig,
    model: Decoder,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    start_epoch: int,
):
    # TODO: Summary writer

    total_bar = tqdm(
        total=config.train.epochs * len(train_loader),
        position=0,
        desc="Steps",
        dynamic_ncols=True,
    )
    total_bar.n = (start_epoch - 1) * len(train_loader)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.train.lr_decay, last_epoch=start_epoch - 2
    )

    interrupted = False

    for epoch in range(start_epoch, config.train.epochs + 1):
        # epoch_loss_dict = None
        for _batch in tqdm(
            train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True, position=1
        ):
            try:
                batch = to_device(_batch, model.device)
                randomized_lengths = random_end_lengths(batch["lengths"] - 1)
                output, masks = model(batch["tokens_ids"][:, :-1], randomized_lengths)
                loss = model.loss(output, batch["tokens_ids"][:, 1:], masks)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if total_bar.n % config.train.logging_steps == 0:
                    tqdm.write(f"Step {total_bar.n}, Loss: {loss.item()}")

                total_bar.update()
            except KeyboardInterrupt:
                interrupted = True
    
        if total_bar.n % config.train.validation_epochs == 0:
            # TODO: Validation
            pass

        if epoch % config.train.save_epochs == 0 or interrupted:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(config.path.log_dir, f"checkpoint_{epoch}.pth")
            )


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default="config/simple_stories.yaml")
    args = parser.parse_args()
    config = SLMConfig.load(args.config)

    # Set random seed
    torch.manual_seed(config.train.manual_seed)
    torch.cuda.manual_seed(config.train.manual_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Decoder(
        config.model.n_layers,
        config.model.d_model,
        config.model.n_heads
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=config.train.betas,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay
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

    tokenizer = tiktoken.get_encoding(config.tokenizer)
    train_dataset = datasets.load_dataset(config.dataset, split="train")
    train_dataset = train_dataset.map(
        lambda x: tokenizer.encode(x["story"]), batched=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=lambda x: dataset_collate(x, torch_convert=True)
    )

    train(config, model, optimizer, train_loader, start_epoch)

if __name__ == "__main__":
    main()
