import matplotlib.pyplot as plt

from sklearn.utils import check_random_state

import torch
from torch import optim

from enflows import transforms, distributions, flows
from enflows.datasets import InfiniteLoader

device = "cpu"

random_state = 42  # Random seed for reproducibility
mean = 0  # Mean of the normal distribution
std_dev = 1  # Standard deviation of the normal distribution
num_samples = 100  # Number of samples to generate
num_features = 2  # Number of features (dimensions)

rng = check_random_state(42)
x = rng.normal(loc=mean, scale=std_dev, size=(num_samples, num_features))

# x, y = datasets.make_moons(128, noise=.1)
plt.scatter(x[:, 0], x[:, 1])
# plt.show()

MB_SIZE = 10
train_loader = InfiniteLoader(
    dataset=x,
    batch_size=MB_SIZE,
    shuffle=True,
    drop_last=True,
    num_epochs=None
)

test_loader = InfiniteLoader(
    dataset=x,
    batch_size=10,
    shuffle=True,
    drop_last=True,
    num_epochs=None
)

transform = transforms.CompositeTransform([
    transforms.ScalarScale(1),
    transforms.ScalarShift(0)
])

base_distribution = distributions.StandardNormal(shape=[2])
flow = flows.Flow(transform=transform, distribution=base_distribution)


def main():
    optimizer = optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
    try:
        for i in range(1000):
            x = next(train_loader).to(device)
            optimizer.zero_grad()
            loss = -flow.log_prob(inputs=x).mean()
            if (i % 50) == 0:
                print(f"{i:04}: {loss=:.3f}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (i + 1) % 250 == 0:
                with torch.no_grad():
                    flow.eval()
                    x = next(test_loader).to(device)
                    test_loss = -flow.log_prob(inputs=x).mean()
                    print(f"{i:04}: {test_loss=:.3f}")
                    flow.train()

    except KeyboardInterrupt:
        pass

    log_prob = flow.log_prob(x)
    samples = flow.sample(2)
    print(log_prob, samples)


if __name__ == "__main__":
    main()
