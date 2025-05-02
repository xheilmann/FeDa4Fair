import argparse
import sys
from collections import OrderedDict
from typing import List, Tuple

import flwr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

# from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader

from fairfl_data.FairFederatedDataset import FairFederatedDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()


# def signal_handler(sig, frame):
#     print("Gracefully stopping your experiment! Keep calm!")
#     global wandb_run
#     if wandb_run:
#         wandb_run.finish()
#     sys.exit(0)


# parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
# parser.add_argument("--dataset_name", type=str, default=None)
# parser.add_argument("--dataset_path", type=str, default=None)
# parser.add_argument("--epochs", type=int, default=None)
# parser.add_argument("--fl_rounds", type=int, default=None)
# parser.add_argument("--batch_size", type=int, default=None)
# parser.add_argument("--optimizer", type=str, default=None)
# parser.add_argument("--sweep", type=bool, default=False)
# parser.add_argument("--cross_device", type=bool, default=False)
# parser.add_argument("--tabular", type=bool, default=False)
# parser.add_argument("--num_nodes", type=int, default=None)
# parser.add_argument("--split_approach", type=str, default=None)
# parser.add_argument("--lr", type=float, default=None)
# parser.add_argument("--seed", type=int, default=None)
# parser.add_argument("--node_shuffle_seed", type=int, default=None)

# parser.add_argument("--alpha_dirichlet", type=float, default=None)
# parser.add_argument("--ratio_unfair_nodes", type=float, default=None)  # number of nodes to make unfair
# parser.add_argument("--group_to_reduce", type=float, default=None, nargs="+")
# parser.add_argument("--group_to_increment", type=float, default=None, nargs="+")
# parser.add_argument(
#     "--ratio_unfairness", type=float, default=None
# )  # how much we want to unbalance the dataset on the unfair nodes
# parser.add_argument("--validation_size", type=float, default=None)
# parser.add_argument("--test_size", type=float, default=None)

# # Parameters for privacy-preserving trainign
# parser.add_argument("--epsilon", type=float, default=None)
# parser.add_argument("--clipping", type=float, default=100000000)

# # Percentage of nodes to use for training, validation and test
# parser.add_argument("--fraction_fit_nodes", type=float, default=None)
# parser.add_argument("--fraction_validation_nodes", type=float, default=None)
# parser.add_argument("--fraction_test_nodes", type=float, default=None)

# # Percentage of nodes to sample for training, validation and test
# parser.add_argument("--sampled_training_nodes", type=float, default=1.0)
# parser.add_argument("--sampled_validation_nodes", type=float, default=0)
# parser.add_argument("--sampled_test_nodes", type=float, default=1.0)

# # Parameters for the wandb logging
# parser.add_argument("--wandb", type=bool, default=False)
# parser.add_argument("--run_name", type=str, default=None)
# parser.add_argument("--project_name", type=str, default=None)

# # configuration for the clients. What percentage of CPU and GPU each client can use
# parser.add_argument("--num_client_cpus", type=float, default=None)
# parser.add_argument("--num_client_gpus", type=float, default=None)

# parser.add_argument("--device", type=str, default="cuda")

# parser.add_argument("--save_local_models", type=bool, default=False)
# parser.add_argument("--save_aggregated_model", type=bool, default=False)


NUM_CLIENTS = 10
BATCH_SIZE = 32


def load_datasets(partition_id: int):
    fds = FairFederatedDataset(
        dataset="ACSIncome",
        states=["CT", "NY"],
        partitioners={
            "CT": 10,
            "NY": 10,
        },
        fl_setting="cross-silo",
        fairness_metric="DP",
        fairness_level="attribute",
    )  # FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(
        split="CT",
        partition_id=partition_id,
    )
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # pytorch_transforms = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )

    # def apply_transforms(batch):
    #     # Instead of passing transforms to CIFAR10(..., transform=transform)
    #     # we will use this function to dataset.with_transform(apply_transforms)
    #     # The transforms object is exactly the same
    #     batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    #     return batch

    # Create train/val for each partition and wrap it into DataLoader
    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader


trainloader, _, _ = load_datasets(partition_id=0)
batch = next(iter(trainloader))
images, labels = batch["img"], batch["label"]

# Reshape and convert images to a NumPy array
# matplotlib requires images with the shape (height, width, 3)
images = images.permute(0, 2, 3, 1).numpy()

# Denormalize
images = images / 2 + 0.5

# Create a figure and a grid of subplots
fig, axs = plt.subplots(4, 8, figsize=(12, 6))

# Loop over the images and plot them
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i])
    ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
    ax.axis("off")

# Show the plot
fig.tight_layout()
plt.show()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(partition_id=partition_id)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, trainloader, valloader).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)

# Create FedAvg strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,
)


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=5)

    return ServerAppComponents(strategy=strategy, config=config)


# Create the ServerApp
server = ServerApp(server_fn=server_fn)

ray_num_cpus = 20
ray_num_gpus = 1
ram_memory = 16_000 * 1024 * 1024 * 2
backend_config = {
    "client_resources": {
        # "include_dashboard": False,
        "num_cpus": ray_num_cpus,
        "num_gpus": ray_num_gpus,
        "memory": ram_memory,
        # "_redis_max_memory": 10000000,
        "object_store_memory": 78643200,
        # "log_to_driver": True,
    }
}


# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)
