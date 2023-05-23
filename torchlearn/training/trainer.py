"""TODO
"""
from typing import Sequence, Type, Union, Optional, Any, Mapping, Dict, Sized

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from torchlearn.metric.metric_dict import MetricDict
from torchlearn.metric.metric_value import MetricValue
from torchlearn.processing.graph import ProcessingGraph, ProcessingFunction
from torchlearn.training.callback import Callback
from torchlearn.training.early_stopping import EarlyStopping
from torchlearn.training.results import Result, EpochResult
from torchlearn.training.schedulers import Scheduler


class Trainer:
    """TODO"""
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

    def train(self, processing_function: ProcessingFunction, epochs: int,
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None, schedulers: Sequence[Scheduler] = (),
              callbacks: Sequence[Callback] = (),
              metrics: Optional[MetricDict | Mapping[str, MetricValue]] = None,
              val_metrics: Optional[MetricDict | Mapping[str, MetricValue]] = None,
              early_stopping: Optional[EarlyStopping] = None,
              trace: bool = False,
              progress: bool = True,
              show_metrics: bool = True) -> Sequence[Result]:
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, MetricDict):
            metrics = MetricDict(**metrics)
        if val_metrics is None:
            val_metrics = metrics
        if not isinstance(val_metrics, MetricDict):
            val_metrics = MetricDict(**val_metrics)
        processing_graph = processing_function(self.model)
        if "loss" not in processing_graph:
            raise ValueError("Missing required 'loss' key in processing graph.")
        if "batch_size" not in processing_graph:
            raise ValueError("Missing required 'batch_size' key in processing graph.")
        train_results = Result()
        valid_results = Result()
        results = (train_results,) if val_dataloader is None else (train_results, valid_results)
        postfix: Dict[str, float] = {}
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

    def training(self, processing_graph: ProcessingGraph, dataloader: DataLoader, metrics: MetricDict,
                 progress: bool = True,
                 show_metrics: bool = True) -> EpochResult:
        optimizer = self.optimizer
        self.model.train()
        epoch_metrics = EpochResult(metrics)
        states = metrics.states
        metrics.reset()
        parameters = metrics.parameters | {"loss", "batch_size"}
        sized = len(dataloader.dataset) if isinstance(dataloader.dataset, Sized) else None
        with tqdm(dataloader, desc="Training", leave=False, disable=not progress,
                  total=sized) as pbar:
            for batch in pbar:
                batch = tuple(t.to(self.device) for t in batch)
                outputs = processing_graph(*parameters, inputs=batch)
                optimizer.zero_grad()
                outputs["loss"].backward()
                optimizer.step()
                for s in states:
                    s(outputs)
                if sized:
                    pbar.update(outputs["batch_size"])
                if show_metrics:
                    pbar.set_postfix(epoch_metrics.values())
        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, processing_graph: ProcessingGraph, dataloader: DataLoader, metrics: MetricDict,
                 progress: bool = True, show_metrics: bool = True) -> EpochResult:
        model = self.model
        model.eval()
        epoch_metrics = EpochResult(metrics)
        states = metrics.states
        metrics.reset()
        parameters = metrics.parameters | {"batch_size"}
        sized = len(dataloader.dataset) if isinstance(dataloader.dataset, Sized) else None
        with tqdm(dataloader, desc="Evaluating", leave=False, disable=not progress,
                  total=sized) as pbar:
            for batch in pbar:
                batch = tuple(t.to(self.device) for t in batch)
                outputs = processing_graph(*parameters, inputs=batch)
                for s in states:
                    s(outputs)
                if sized:
                    pbar.update(outputs["batch_size"])
                if show_metrics:
                    pbar.set_postfix(epoch_metrics.values())
        return epoch_metrics
