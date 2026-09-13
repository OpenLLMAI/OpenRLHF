import copy
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader


def _load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sft_module(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("_sft_loss_test")
    package.__path__ = [str(root / "openrlhf/models")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    _load_module(monkeypatch, "_sft_loss_test.utils", root / "openrlhf/models/utils.py")
    loss = _load_module(monkeypatch, "_sft_loss_test.loss", root / "openrlhf/models/loss.py")
    monkeypatch.setitem(sys.modules, "openrlhf.models", SimpleNamespace(SFTLoss=loss.SFTLoss))
    module = _load_module(monkeypatch, "_sft_trainer_test", root / "openrlhf/trainer/sft_trainer.py")
    monkeypatch.setattr(module, "tqdm", MagicMock())
    return module


class _Engine(torch.nn.Module):
    """CPU optimizer with DeepSpeed's managed accumulation and boundary override."""

    def __init__(self, gas):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.gas = gas
        self.micro_steps = 0
        self.global_steps = 0
        self.global_samples = 0
        self.batch_size = gas
        self.boundary = None
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        self.gradients = []

    def set_gradient_accumulation_boundary(self, boundary):
        self.boundary = boundary

    def backward(self, loss):
        (loss / self.gas).backward()

    def step(self):
        boundary = (self.micro_steps + 1) % self.gas == 0 if self.boundary is None else self.boundary
        if boundary:
            self.gradients.append(self.weight.grad.item())
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_steps += 1
            self.global_samples += self.batch_size
        self.micro_steps += 1


class _Actor(torch.nn.Module):
    def __init__(self, gas):
        super().__init__()
        self.model = _Engine(gas)
        self.seen = []

    def forward(self, inputs, **kwargs):
        self.seen.extend(inputs[:, 0].tolist())
        logps = -self.model.weight * inputs[:, 1:].float()
        return logps, SimpleNamespace(aux_loss=2 * self.model.weight)


class _Strategy:
    ring_attn_group = None

    def __init__(self, gas):
        self.accumulated_gradient = gas

    def is_rank_0(self):
        return False

    def all_reduce(self, value):
        return value

    def backward(self, loss, actor, optimizer):
        actor.model.backward(loss)

    def optimizer_step(self, optimizer, actor, scheduler):
        actor.model.step()

    def get_grad_norm(self, actor):
        return 0.0


def _collate(samples):
    inputs = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True)
    attention = torch.nn.utils.rnn.pad_sequence([torch.ones_like(s) for s in samples], batch_first=True)
    masks = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor([1.0] * (len(s) - 1) + [0.0]) for s in samples], batch_first=True
    )
    return inputs, attention, masks


def _make_trainer(module, lengths, gas, epochs=2, aux_coef=0.0, shuffle=False, micro_batch_size=1):
    # First token identifies the sample; remaining tokens are differentiable NLL coefficients.
    samples = [torch.tensor([i + 1] + [2 * i + 1] * length) for i, length in enumerate(lengths)]
    sampler = module.DistributedSampler(samples, num_replicas=1, rank=0, shuffle=shuffle, seed=42, drop_last=True)
    loader = DataLoader(samples, batch_size=micro_batch_size, sampler=sampler, collate_fn=_collate, drop_last=True)
    trainer = module.SFTTrainer.__new__(module.SFTTrainer)
    trainer.epochs = epochs
    trainer.strategy = _Strategy(gas)
    trainer.model = _Actor(gas)
    trainer.model.model.batch_size = gas * micro_batch_size
    trainer.optimizer = trainer.model.model.optimizer
    trainer.scheduler = trainer.model.model.scheduler
    trainer.train_dataloader = loader
    trainer.loss_fn = module.SFTLoss()
    trainer.aux_loss = aux_coef > 0
    trainer._wandb = trainer._tensorboard = None
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(batch_size=gas * micro_batch_size),
        model=SimpleNamespace(aux_loss_coef=aux_coef),
        eval=SimpleNamespace(steps=-1),
        ckpt=SimpleNamespace(save_steps=-1),
    )
    trainer.checkpoints = []

    def save(args, global_step, bar, logs, client_states):
        engine = trainer.model.model
        trainer.checkpoints.append(
            copy.deepcopy(
                {
                    "global_step": global_step,
                    "global_samples": engine.global_samples,
                    "client_states": client_states,
                    "model": engine.state_dict(),
                    "optimizer": engine.optimizer.state_dict(),
                    "scheduler": engine.scheduler.state_dict(),
                    "seen": trainer.model.seen[:],
                    "logs": logs,
                }
            )
        )

    trainer.save_logs_and_checkpoints = save
    return trainer


def _reference(trainer, epochs):
    loader = trainer.train_dataloader
    gradients, seen, batches = [], [], []
    for epoch in range(epochs):
        loader.sampler.set_epoch(epoch)
        batches.extend(loader)
    gas = trainer.strategy.accumulated_gradient
    for start in range(0, len(batches) // gas * gas, gas):
        total, count = 0.0, 0.0
        for inputs, _, mask in batches[start : start + gas]:
            seen.extend(inputs[:, 0].tolist())
            total += (inputs[:, 1:] * mask[:, :-1]).sum().item()
            count += mask[:, :-1].sum().item()
        gradients.append(total / count + 2 * trainer.args.model.aux_loss_coef)
    loader.sampler.set_epoch(0)
    return gradients, seen


@pytest.mark.parametrize(
    "lengths,gas,micro_batch_size",
    [
        ([1, 1, 1, 1, 1], 4, 1),
        ([1, 3, 2, 5, 4], 4, 1),
        ([1, 3, 2], 4, 1),
        ([1, 3, 2], 8, 1),
        ([1, 3, 2, 5], 2, 1),
        ([1, 3, 2, 5, 4], 1, 1),
        ([1, 3, 2, 5, 4, 1, 2, 3, 1, 4], 4, 2),
    ],
)
@pytest.mark.parametrize("aux_coef", [0.0, 0.1])
def test_epoch_updates_match_token_mean_reference(sft_module, lengths, gas, micro_batch_size, aux_coef):
    trainer = _make_trainer(sft_module, lengths, gas, epochs=3, aux_coef=aux_coef, micro_batch_size=micro_batch_size)
    expected, seen = _reference(trainer, 3)
    trainer.fit(trainer.args)
    engine = trainer.model.model
    assert engine.gradients == pytest.approx(expected)
    assert trainer.model.seen == seen
    assert engine.global_steps == len(expected)
    assert engine.scheduler.last_epoch == len(expected)
    assert engine.weight.grad is None
    assert [c["global_step"] for c in trainer.checkpoints] == list(range(1, len(expected) + 1))
    assert engine.global_samples == len(seen)
    assert all(c["global_samples"] == c["client_states"]["consumed_samples"] for c in trainer.checkpoints)
    assert trainer.args.eval.steps == max(1, len(trainer.train_dataloader) // gas)
    assert engine.boundary is None


@pytest.mark.parametrize("checkpoint_index", range(3))
@pytest.mark.parametrize("micro_batch_size", [1, 2])
def test_resume_matches_uninterrupted_training(sft_module, checkpoint_index, micro_batch_size):
    lengths = [1, 3, 2, 5, 4] if micro_batch_size == 1 else [1, 3, 2, 5, 4] * 2 + [1]
    trainer = _make_trainer(sft_module, lengths, 4, epochs=3, shuffle=True, micro_batch_size=micro_batch_size)
    trainer.fit(trainer.args)
    state = trainer.checkpoints[checkpoint_index]
    resumed = _make_trainer(sft_module, lengths, 4, epochs=3, shuffle=True, micro_batch_size=micro_batch_size)
    engine = resumed.model.model
    engine.load_state_dict(state["model"])
    engine.optimizer.load_state_dict(state["optimizer"])
    engine.scheduler.load_state_dict(state["scheduler"])
    engine.global_steps = state["global_step"]
    engine.global_samples = state["global_samples"]
    resumed.fit(resumed.args, consumed_samples=state["client_states"]["consumed_samples"])
    assert torch.equal(engine.weight, trainer.model.model.weight)
    assert engine.global_steps == trainer.model.model.global_steps
    assert engine.global_samples == trainer.model.model.global_samples
    assert engine.scheduler.state_dict() == trainer.model.model.scheduler.state_dict()
    assert state["seen"] + resumed.model.seen == trainer.model.seen
    assert [c["global_step"] for c in resumed.checkpoints] == list(range(state["global_step"] + 1, 4))


@pytest.mark.parametrize(
    "dataset_size,micro_batches,gas,expected_steps", [(5, 5, 4, 2), (3, 3, 4, 1), (8, 4, 2, 4), (7, 7, 4, 3)]
)
def test_cli_scheduler_uses_actual_optimizer_windows(
    monkeypatch, tmp_path, dataset_size, micro_batches, gas, expected_steps
):
    root = Path(__file__).resolve().parents[1]
    dataset = MagicMock()
    dataset.__len__.return_value = dataset_size
    loader = MagicMock()
    loader.__len__.return_value = micro_batches
    strategy = MagicMock(accumulated_gradient=gas)
    strategy.setup_dataloader.return_value = loader
    strategy.prepare.return_value = (MagicMock(), MagicMock(), MagicMock())
    for name in [
        "openrlhf.datasets",
        "openrlhf.datasets.utils",
        "openrlhf.models",
        "openrlhf.trainer.sft_trainer",
        "openrlhf.utils",
    ]:
        monkeypatch.setitem(sys.modules, name, MagicMock())
    module = _load_module(monkeypatch, "_sft_cli_test", root / "openrlhf/cli/train_sft.py")
    module.get_strategy.return_value = strategy
    module.SFTDataset.return_value = dataset
    args = MagicMock()
    args.model.gradient_checkpointing_enable = False
    args.eval.dataset = None
    args.ckpt.load_enable = False
    args.ckpt.output_dir = str(tmp_path)
    args.train.max_epochs = 2
    args.train.batch_size = 4
    args.data.max_samples = dataset_size
    module.train(args)
    config = strategy.prepare.call_args.args[0][1]
    assert config["scheduler_steps"] == expected_steps
    assert module.SFTTrainer.return_value.fit.call_args.args[2] == max(1, micro_batches // gas)


@pytest.mark.parametrize("lengths,epochs", [([], 2), ([1, 2, 3], 1)])
def test_no_complete_update_is_rejected(sft_module, lengths, epochs):
    trainer = _make_trainer(sft_module, lengths, 4, epochs=epochs)
    with pytest.raises(ValueError, match="complete gradient-accumulation window"):
        trainer.fit(trainer.args)
