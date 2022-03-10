import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DensityEstimationTarget:
    def __init__(self, choice, num_samples):
        self.choices = ['Two circles','Moons','S Curve', 'Hollow Circle', 'Dimension 1', 'Orbits']
        self.choice = choice
        assert self.choice in self.choices, "'" + choice + "'" + ' not implemented, please select from ' + str(
            self.choices)

        if choice == 'Two circles':
            self.p = 2
            self.target_log_density = None
            X, y = datasets.make_circles(num_samples, factor=0.5, noise=0.05)
            X = StandardScaler().fit_transform(X)
            self.target_samples = torch.tensor(X).float()

        if choice == 'Moons':
            self.p = 2
            self.target_log_density = None
            X, y = datasets.make_moons(num_samples, noise=0.05)
            X = StandardScaler().fit_transform(X)
            self.target_samples = torch.tensor(X).float()

        if choice == 'S Curve':
            self.p = 2
            self.target_log_density = None
            X, t = datasets.make_s_curve(num_samples, noise=0.05)
            X = StandardScaler().fit_transform(X)
            self.target_samples = torch.tensor(X[:,[0,2]]).float()

        if choice == "Hollow Circle":
            self.p = 2
            epsilon = 0.01
            theta = 2*3.1415*torch.rand([num_samples])
            samples = torch.cat([torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)], dim = -1)
            samples += torch.randn(samples.shape)*epsilon
            self.target_samples = samples

        if choice == "Dimension 1":
            self.p = 1
            num_component = 6
            means = torch.tensor([[-0.25], [1.875], [4.125], [6.25], [-5.5],[-8.5]])
            covs = torch.tensor([[[1.]], [[.5]], [[.5]], [[2.]], [[1]],[[1]]])
            comp = torch.ones(num_component)
            mvn_target = MultivariateNormal(means, covs)
            cat = Categorical(comp / torch.sum(comp))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])

        if choice == "Orbits":
            self.p = 2
            number_planets = 7
            covs_target = 0.04*torch.eye(self.p).unsqueeze(0).repeat(number_planets,1,1)
            means_target = 2.5*torch.view_as_real(torch.pow(torch.exp(torch.tensor([2j * 3.1415 / number_planets])), torch.arange(0, number_planets)))
            weights_target = torch.ones(number_planets)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_samples = mix_target.sample([num_samples])

    def get_samples(self):
        return self.target_samples

    def target_visual(self, num_samples = 5000):
        num_samples = min(num_samples, self.target_samples.shape[0])
        if self.p == 1:
            plt.figure(figsize=(10, 5))
            sns.histplot(self.target_samples[:num_samples][:, 0].cpu(), bins=150, color='red', stat='density', alpha=0.6,label=self.choice + " samples")
            plt.legend()

        if self.p == 2:
            plt.figure(figsize=(10, 5))
            plt.scatterplot(self.target_samples[:, 0], self.target_samples[:,1],color='red', stat='density', alpha=0.6,label=self.choice + " samples")
            plt.legend()
        Z
