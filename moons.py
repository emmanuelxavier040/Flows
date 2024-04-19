import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import optim

from enflows.flows.base import Flow
from enflows.distributions.normal import StandardNormal
from enflows.transforms.base import CompositeTransform
from enflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from enflows.transforms.permutations import ReversePermutation

x, y = datasets.make_moons(128, noise=.1)
plt.scatter(x[:, 0], x[:, 1])
plt.savefig("./figures/original.png")

num_layers = 5
base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2,
                                                          hidden_features=4))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

num_iter = 25000
for i in range(num_iter):
    x, y = datasets.make_moons(128, noise=.1)
    x = torch.tensor(x, dtype=torch.float32)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
        xline = torch.linspace(-1.5, 2.5,steps=100)
        yline = torch.linspace(-.75, 1.25,steps=100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        plt.title('iteration {}'.format(i + 1))
        filename = "./figures/sine_wave_plot"+str(i)+".png"
        plt.savefig(filename)
        # plt.show()

