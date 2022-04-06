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
            plt.legend()

        if samples.shape[-1] >= 2:
            plt.figure(figsize=(10, 5))
            plt.scatter(samples[:, 0], samples[:,1],color='red',alpha=0.6)
            plt.legend()

class TwoCircles(DensityEstimationTarget):
    def __init__(self):
        pass

    def sample(self, num_samples):
        X, y = datasets.make_circles(num_samples, factor=0.5, noise=0.05)
        X = StandardScaler().fit_transform(X)
        return torch.tensor(X).float()

class Moons(DensityEstimationTarget):
    def __init__(self):
        pass

    def sample(self, num_samples):
        X, y = datasets.make_moons(num_samples, noise=0.05)
        X = StandardScaler().fit_transform(X)
        return torch.tensor(X).float()

class SCurve(DensityEstimationTarget):
    def __init__(self):
        pass

    def sample(self, num_samples):
        X, t = datasets.make_s_curve(num_samples, noise=0.05)
        X = StandardScaler().fit_transform(X)
        return torch.tensor(X).float()

class Dimension1(DensityEstimationTarget):
    def __init__(self):
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



    def __init__(self, choice, num_samples):
        self.choices = ['Two circles','Moons','S Curve', 'Dimension 1', 'Orbits']
        self.choice = choice
        assert self.choice in self.choices, "'" + choice + "'" + ' not implemented, please select from ' + str(
            self.choices)
        if choice == 'S Curve':
            X, t = datasets.make_s_curve(num_samples, noise=0.05)
            X = StandardScaler().fit_transform(X)
            self.target_samples = torch.tensor(X[:,[0,2]]).float()

        if choice == "Dimension 1":
            num_component = 6
            means = torch.tensor([[-0.25], [1.875], [4.125], [6.25], [-5.5],[-8.5]])
            covs = torch.tensor([[[1.]], [[.5]], [[.5]], [[2.]], [[1]],[[1]]])
            comp = torch.ones(num_component)
            mvn_target = MultivariateNormal(means, covs)
            cat = Categorical(comp / torch.sum(comp))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])

        if choice == "Orbits":
            number_planets = 7
            covs_target = 0.04*torch.eye(2).unsqueeze(0).repeat(number_planets,1,1)
            means_target = 2.5*torch.view_as_real(torch.pow(torch.exp(torch.tensor([2j * 3.1415 / number_planets])), torch.arange(0, number_planets)))
            weights_target = torch.ones(number_planets)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])

        if choice == "Banana":
            self.var = 2
            self.dim = 50
            self.even = torch.arange(0,self.dim, 2)
            self.odd = torch.arange(1,self.dim, 2)
            self.mvn = torch.distributions.MultivariateNormal(torch.zeros(self.dim), self.var * torch.eye(self.dim))
            x = self.mvn.sample([num_samples])
            z = x.clone()
            z[..., self.odd] += z[..., self.even] ** 2
            self.target_samples = z


    def get_samples(self):
        return self.target_samples

    def target_visual(self, num_samples = 5000):
        num_samples = min(num_samples, self.target_samples.shape[0])
        if self.target_samples.shape[-1] == 1:
            plt.figure(figsize=(10, 5))
            plt.hist(self.target_samples[:num_samples][:, 0].cpu().numpy(), bins=150, color='red',density = True, alpha=0.6,label=self.choice + " samples")
            plt.legend()

        if self.target_samples.shape[-1] >= 2:
            plt.figure(figsize=(10, 5))
            plt.scatter(self.target_samples[:, 0], self.target_samples[:,1],color='red',alpha=0.6,label=self.choice + " samples")
            plt.legend()

class Banana(DensityEstimationTarget):
    def __init__(self):
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

