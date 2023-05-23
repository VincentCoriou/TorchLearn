# TorchLearn

TorchLearn is a Python package that provides an additional layer on top of vanilla pytorch
in order to simplify and fasten the development and deployment of Machine Learning Training
pipelines.

TorchLearn defines a `Trainer` abstraction that allows to define the processing logic as a 
Directed Acyclic Graph (DAG) and only execute the nodes that are necessary to compute the 
loss and metrics. An example of DAG definition and training is shown below:

```
from functools import partial

from torch.nn import functional as F
from torch.optim import Adam

from torchlearn.processing.graph import ProcessingGraph
from torchlearn.metric.classification import ConfusionMatrix, Accuracy, Precision

def processing(model):
    return ProcessingGraph(
        inputs=("input", "target", "sample_weight")
        functions=(
            ("output", model, "input"),
            ("predicted", partial(torch.argmax, dim=-1), "output"),
            ("sample_loss", partial(F.cross_entropy, reduction="none"), ("output", "target")),
            ("weighted_loss", torch.prod, ("sample_loss", "sample_weight")),
            ("loss", torch.sum, "weighted_loss")
        )
)

cm = ConfusionMatrix(classes)
metrics = {"acc": Accuracy(cm), "precision": Precision(cm, 1)}

trainer = Trainer(model, Adam)
results = trainer.train(processing, epochs, train_loader, val_loader, metrics=metrics)
```