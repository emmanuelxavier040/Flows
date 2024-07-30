import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import os
import torch
from torch import optim

from enflows.flows.base import Flow
from enflows.distributions.normal import StandardNormal
from enflows.transforms.base import CompositeTransform
from enflows.transforms.linear import NaiveLinear, ScalarScale, ScalarShift


base_directory = "./figures/LinearFlows"
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

x, y = datasets.make_moons(128, noise=.1)
plt.scatter(x[:, 0], x[:, 1])
plt.savefig(base_directory + "/target_distribution.png")

num_layers = 5
base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    transforms.append(NaiveLinear(features=2))
    transforms.append(ScalarScale(scale=2))
    transforms.append(ScalarShift(shift=1.5))

transform = CompositeTransform(transforms)
flow = Flow(transform, base_dist)

optimizer = optim.Adam(flow.parameters())
num_iter = 10000
for i in range(num_iter):
    x, y = datasets.make_moons(128, noise=.1)
    x = torch.tensor(x, dtype=torch.float32)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()

    if (i + 1) % 1000 == 0:
        xline = torch.linspace(-1.5, 2.5, steps=100)
        yline = torch.linspace(-.75, 1.25, steps=100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        plt.title('iteration {}'.format(i + 1))
        filename = base_directory + "/approximate_distribution_plot" + str(i) + ".png"
        plt.savefig(filename)
        # plt.show()
