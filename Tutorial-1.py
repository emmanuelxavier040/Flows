import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma

d = 2
k = 1
epochs = 1000

mu, cov = torch.zeros(2), torch.eye(2)
base_dist = MultivariateNormal(mu, cov)

shape_params = torch.tensor([2.0, 3.0])
rate_params = torch.tensor([1.0, 1.5])
gamma_dist = Gamma(shape_params, rate_params)
num_samples = torch.Size([1000])
x = gamma_dist.sample(num_samples)
plt.scatter(x[:, 0], x[:, 1])
plt.show()


class Flows(nn.Module):
    def __init__(self, d, k, hidden):
        super().__init__()
        self.d, self.k = d, k
        self.sig_net = nn.Sequential(
            nn.Linear(k, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, d - k))

        self.mu_net = nn.Sequential(
            nn.Linear(k, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, d - k))

    def forward(self, x):
        x1, x2 = x[:, :self.k], x[:, self.k:]
        sig = self.sig_net(x1)
        mu = self.mu_net(x1)
        z1 = x1
        z2 = x2 * torch.exp(sig) + mu
        z = torch.cat([z1, z2], dim=-1)
        log_pz = base_dist.log_prob(z)
        log_jacob = sig.sum(-1)
        return z, log_pz, log_jacob

    def inverse(self, z):
        z1, z2 = z[:, :self.k], z[:, self.k:]
        sig = self.sig_net(z1)
        mu = self.mu_net(z1)
        x1 = z1
        x2 = (z2 - mu) * torch.exp(-sig)
        x = torch.concat([x1, x2], dim=-1)
        return x

    def log_prob(self, x):
        return base_dist.log_prob(x)


model = Flows(d, k, hidden=512)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)

losses = []
for i in range(epochs):
    optimizer.zero_grad()
    x = gamma_dist.sample(num_samples)
    z, log_pz, log_jacob = model.forward(x)

    # maximize p_X(x) == minimize -p_X(x)
    loss = -(log_jacob + log_pz).mean()
    losses.append(loss)

    loss.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
        print(i)
        xline = torch.linspace(-1.5, 2.5, steps=100)
        yline = torch.linspace(-.75, 1.25, steps=100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = model.log_prob(xyinput).exp().reshape(100, 100)

        plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        plt.title('iteration {}'.format(i + 1))
        filename = "./figures/sine_wave_plot" + str(i) + ".png"
        # print(xyinput, model.log_prob(xyinput))
        plt.savefig(filename)

losses = [loss.detach().numpy() for loss in losses]
plt.plot(losses)
plt.title("Model Loss vs Epoch")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
# x_gen = model.inverse(z)
# print(x_gen)
# x_gen = x_gen.detach().numpy()
# plt.scatter(x_gen[:, 0], x_gen[:, 1])
# plt.show()
