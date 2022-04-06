import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from torch import nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class DensityEstimationTarget(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self):
        raise NotImplementedError

    def target_visual(self, num_samples = 5000):
        samples = self.sample(num_samples)
        if samples.shape[-1] == 1:
            plt.figure(figsize=(10, 5))
            plt.hist(samples[:, 0].numpy(), bins=150, color='red',density = True, alpha=0.6)

        if samples.shape[-1] >= 2:
            plt.figure(figsize=(10, 5))
            plt.scatter(samples[:,-2], samples[:,-1],color='red',alpha=0.6)

class TwoCircles(DensityEstimationTarget):
    def __init__(self):
        super().__init__()

    def sample(self, num_samples):
        X, y = datasets.make_circles(num_samples, factor=0.5, noise=0.05)
        X = StandardScaler().fit_transform(X)
        return torch.tensor(X).float()

class Moons(DensityEstimationTarget):
    def __init__(self):
        super().__init__()

    def sample(self, num_samples):
        X, y = datasets.make_moons(num_samples, noise=0.05)
        X = StandardScaler().fit_transform(X)
        return torch.tensor(X).float()

class SCurve(DensityEstimationTarget):
    def __init__(self):
        super().__init__()

    def sample(self, num_samples):
        X, t = datasets.make_s_curve(num_samples, noise=0.05)
        X = StandardScaler().fit_transform(X)
        return torch.tensor(X).float()

class Dimension1(DensityEstimationTarget):
    def __init__(self):
        super().__init__()
        num_component = 6
        means = torch.tensor([[-0.25], [1.875], [4.125], [6.25], [-5.5], [-8.5]])
        covs = torch.tensor([[[1.]], [[.5]], [[.5]], [[2.]], [[1]], [[1]]])
        comp = torch.ones(num_component)
        mvn_target = MultivariateNormal(means, covs)
        cat = Categorical(comp / torch.sum(comp))
        self.mix_target = MixtureSameFamily(cat, mvn_target)

    def sample(self, num_samples):
        return self.mix_target.sample([num_samples])

class Orbits(DensityEstimationTarget):
    def __init__(self):
        super().__init__()
        number_planets = 7
        covs_target = 0.04 * torch.eye(2).unsqueeze(0).repeat(number_planets, 1, 1)
        means_target = 2.5 * torch.view_as_real(
            torch.pow(torch.exp(torch.tensor([2j * 3.1415 / number_planets])), torch.arange(0, number_planets)))
        weights_target = torch.ones(number_planets)
        weights_target = weights_target

        mvn_target = MultivariateNormal(means_target, covs_target)
        cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
        self.mix_target = MixtureSameFamily(cat, mvn_target)

    def sample(self, num_samples):
        return self.mix_target.sample([num_samples])

class Banana(DensityEstimationTarget):
    def __init__(self):
        super().__init__()
        var = 2
        dim = 50
        self.even = torch.arange(0, dim, 2)
        self.odd = torch.arange(1, dim, 2)
        self.mvn = torch.distributions.MultivariateNormal(torch.zeros(dim), var * torch.eye(dim))

    def transform(self, x):
        z = x.clone()
        z[...,self.odd] += z[...,self.even]**2
        return z

    def sample(self, num_samples):
        return self.transform(self.mvn.sample([num_samples]))

