import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

DEVICE = (
    "gpu"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
EPOCH = 10
LR = 0.05
BATCH_SIZE = 32

train_data = datasets.FashionMNIST(
    root="PyTorch/dataset",
    train=True,
    download=True,
    transform=ToTensor(),
)


test_data = datasets.FashionMNIST(
    root="PyTorch/dataset",
    train=False,
    download=True,
    transform=ToTensor(),
)


train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)


class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, X):
        X = self.flatten.forward(X)
        output = self.layers.forward(X)
        return output


def train_loop(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
):
    size = len(dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model.forward(X)
        loss = loss_fn.forward(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: nn.CrossEntropyLoss,
):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred: torch.Tensor = model.forward(X)
            test_loss += loss_fn.forward(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


print(model.state_dict())
for p in list(model.parameters()):
    print(p.shape)
