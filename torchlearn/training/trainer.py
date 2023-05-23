from functools import reduce
from operator import or_
from typing import Sequence, Type, Union, Optional, Any, Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from torchlearn.processing.graph import ProcessingGraph
from torchlearn.training.callback import Callback
from torchlearn.training.results import Result, EpochResult
from torchlearn.training.schedulers import Scheduler


class Trainer:
    device: Optional[torch.device]
    model: nn.Module
    optimizer: Optimizer

    def __init__(self, model: nn.Module, optimizer: Union[Type[Optimizer], Optimizer], *args: Any,
                 device: Optional[torch.device] = None, **kwargs: Any) -> None:
        if device is not None:
            model = model.to(device)
        self.device = device
        self.model = model
        if isinstance(optimizer, type):
            optimizer = optimizer(model.parameters(), *args, **kwargs)
        self.optimizer = optimizer

    def train(self, processing_function: Callable[[nn.Module, ...], ProcessingGraph], epochs: int,
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None, schedulers: Sequence[Scheduler] = (),
              callbacks: Sequence[Callback] = (),
              metrics=None,
              val_metrics=None,
              early_stopping=None,
              trace=False,
              progress=True,
              show_metrics=True):
        if metrics is None:
            metrics = {}
        if val_metrics is None:
            val_metrics = metrics
        processing_graph = processing_function(self.model)
        if "loss" not in processing_graph:
            raise ValueError("Missing required 'loss' key in processing graph.")
        if "batch_size" not in processing_graph:
            raise ValueError("Missing required 'batch_size' key in processing graph.")
        train_results = Result()
        valid_results = Result()
        results = (train_results,) if val_dataloader is None else (train_results, valid_results)
        postfix = {}
        with tqdm(range(1, epochs + 1), leave=trace, desc="Epochs", postfix=postfix, disable=not progress) as pbar:
            try:
                for epoch in pbar:
                    train_result = self.training(processing_graph, train_dataloader, metrics, progress, show_metrics)
                    train_results.add(epoch, train_result)
                    if show_metrics:
                        postfix.update(train_result.values())
                        pbar.set_postfix(ordered_dict=postfix)
                    if val_dataloader is not None:
                        valid_result = self.evaluate(processing_graph, val_dataloader, val_metrics, progress,
                                                     show_metrics)
                        valid_results.add(epoch, valid_result)
                        if show_metrics:
                            postfix.update(valid_result.values("val_"))
                            pbar.set_postfix(ordered_dict=postfix)
                    for scheduler in schedulers:
                        scheduler.step(self.model, epoch)
                    for callback in callbacks:
                        callback(self.model, epoch, results)
                    if early_stopping is not None and early_stopping.step(epoch, *results):
                        pbar.set_description(f"Early stopping (Epoch {epoch}/{epochs})")
                        break
            except KeyboardInterrupt:
                pass
        return results

    def training(self, processing_graph: ProcessingGraph, dataloader: DataLoader, metrics, progress: bool = True,
                 show_metrics: bool = True):
        optimizer = self.optimizer
        self.model.train()
        epoch_metrics = EpochResult(metrics)
        states = reduce(or_, (set(m.states()) for m in metrics.values()), set())
        for s in states:
            s.reset()
        arguments = reduce(or_, (set(m.arguments()) for m in metrics.values()), {"loss"})
        with tqdm(dataloader, desc="Training", leave=False, disable=not progress,
                  total=len(dataloader.dataset)) as pbar:
            for batch in pbar:
                batch = tuple(t.to(self.device) for t in batch)
                outputs = processing_graph(*arguments, inputs=batch)
                optimizer.zero_grad()
                outputs["loss"].backward()
                optimizer.step()
                for s in states:
                    s(outputs)
                pbar.update(outputs["batch_size"])
                if show_metrics:
                    pbar.set_postfix(**epoch_metrics.values())
        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, criterions, dataloader, metrics, progress=True, show_metrics=True):
        model = self.model
        model.eval()
        if not isinstance(criterions, Sequence):
            criterions = (criterions,)
        epoch_metrics = EpochResult(metrics)
        states = reduce(or_, (set(m.states()) for m in metrics.values()), set())
        for s in states:
            s.reset()
        with tqdm(dataloader, desc="Evaluating", leave=False, disable=not progress) as pbar:
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_size = inputs.size(0)
                outputs = model(inputs)
                losses = torch.stack(tuple(criterion(outputs, targets) for criterion in criterions), 0)
                loss = losses.sum()
                args = {"loss": loss, "outputs": outputs[0], "targets": targets, "inputs": inputs, "losses": losses,
                        "batch_size": batch_size}
                for s in states:
                    s(args)
                if show_metrics:
                    pbar.set_postfix(**epoch_metrics.values())
        return epoch_metrics
