import os
import tempfile
import torch
from torch import nn, optim
from utils.checkpoint import load_checkpoint, save_checkpoint, latest_checkpoint_path


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


def test_save_and_load_checkpoint():
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epoch = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "model_5.pt")
        save_checkpoint(model, optimizer, epoch, checkpoint_path, printf=lambda x: None)

        new_model = SimpleModel()
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)
        loaded_epoch = load_checkpoint(checkpoint_path, new_model, new_optimizer, printf=lambda x: None)

        assert loaded_epoch == epoch

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            torch.testing.assert_close(p1, p2)


def test_load_checkpoint_without_optimizer():
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epoch = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "model_3.pt")
        save_checkpoint(model, optimizer, epoch, checkpoint_path, printf=lambda x: None)

        new_model = SimpleModel()
        loaded_epoch = load_checkpoint(checkpoint_path, new_model, printf=lambda x: None)

        assert loaded_epoch == epoch


def test_latest_checkpoint_path_finds_latest():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple checkpoint files
        for i in [1, 5, 3, 10, 2]:
            path = os.path.join(tmpdir, f"model_{i}.pt")
            torch.save({"epoch": i}, path)

        latest = latest_checkpoint_path(tmpdir, "model_*.pt")

        assert latest is not None
        assert latest.endswith("model_10.pt")


def test_latest_checkpoint_path_returns_none_when_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        latest = latest_checkpoint_path(tmpdir, "model_*.pt")

        assert latest is None


def test_save_checkpoint_creates_file():
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "model.pt")
        save_checkpoint(model, optimizer, 1, checkpoint_path, printf=lambda x: None)

        assert os.path.isfile(checkpoint_path)


def test_checkpoint_preserves_optimizer_state():
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Run a training step to update optimizer state
    x = torch.randn(2, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    original_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in optimizer.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "model.pt")
        save_checkpoint(model, optimizer, 1, checkpoint_path, printf=lambda x: None)

        new_model = SimpleModel()
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)
        load_checkpoint(checkpoint_path, new_model, new_optimizer, printf=lambda x: None)

        # Compare optimizer states (param_groups)
        assert len(new_optimizer.state_dict()["param_groups"]) == len(original_state["param_groups"])
