import torch
import capslearn.torch.optimizer as opt

num_data = 1000
num_epochs = 10

x = torch.randn(num_data, 1)
y = (x ** 2) + 3

net = torch.nn.Sequential(
        torch.nn.Linear(1, 6),
        torch.nn.ReLU(),
        torch.nn.Linear(6, 1),
        )

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
optimizer = opt.TestOptimizer(optimizer)

losses = []

for i in range(num_epochs):
    optimizer.zero_grad()
    output = net(x)
    loss = loss_func(output, y)
    loss.backward()

    optimizer.step()
    losses.append(loss.item())
