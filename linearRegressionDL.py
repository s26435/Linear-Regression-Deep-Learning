import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# hyperconstants
learning_rate = .0001
epochs = 20000
randomizer_seed = 42
data_type = torch.float32

# create train /test sets
weight = .7
bias = .3
start = 0
end = 1
step = .02
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * x + bias
trainSplit = int(0.8 * len(x))
X_train, Y_train = x[:trainSplit], y[:trainSplit]
X_test, Y_test = x[trainSplit:], y[trainSplit:]


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=data_type))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=data_type))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


torch.manual_seed(randomizer_seed)
model_0 = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model_0(X_test)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=learning_rate)

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, Y_test)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
print(model_0.state_dict())
print(f"Should be weight = {weight} , bias = {bias}")
