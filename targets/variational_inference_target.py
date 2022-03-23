import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily, Uniform
import math


class Uniform:
    def __init__(self, lower, upper):
        self.p = upper.shape[0]
        self.lower = lower
        self.upper = upper
        assert torch.sum(upper>lower) == self.p, 'upper bound should be greater or equal to lower bound'
        self.log_scale = torch.log(self.upper - self.lower)
        self.location = (self.upper + self.lower)/2

    def log_prob(self, samples):
        condition = ((samples > self.lower).sum(-1) == self.p) * ((samples < self.upper).sum(-1) == self.p)*1
        inverse_condition = torch.logical_not(condition) * 1
        true = -torch.logsumexp(self.log_scale, dim = -1) * condition
        false = torch.nan_to_num(-torch.inf*inverse_condition, nan = 0)
        return (true + false)

    def sample(self, num_samples):
        desired_size = num_samples.copy()
        desired_size.append(self.p)
        return self.lower.expand(desired_size) + torch.rand(desired_size)*torch.exp(self.log_scale.expand(desired_size))

class Mixture:
    def __init__(self, distributions, weights):
        self.distributions = distributions
        self.number_components = len(self.distributions)
        assert weights.shape[0] == self.number_components, 'wrong number of weights'
        self.weights = weights/torch.sum(weights)

    def log_prob(self, samples):
        list_evaluated_distributions = []
        for i,distribution in enumerate(self.distributions):
            list_evaluated_distributions.append(distribution.log_prob(samples).reshape(samples.shape[:-1]).unsqueeze(1) + torch.log(self.weights[i]))
        return(torch.logsumexp(torch.cat(list_evaluated_distributions, dim =1), dim = 1))

    def sample(self, num_samples):
        sampled_distributions = []
        for distribution in self.distributions:
            sampled_distributions.append(distribution.sample(num_samples).unsqueeze(1))
        temp = torch.cat(sampled_distributions, dim = 1)
        pick = Categorical(self.weights).sample(num_samples).squeeze(-1)
        temp2 = torch.stack([temp[i,pick[i],:] for i in range(temp.shape[0])])
        return temp2


class VariationalInferenceTarget:
    def __init__(self, choice):
        self.choices = ["Sharp Edges", "Dimension 1", "Blob Dimension 64", "Orbits"]
        self.choice = choice
        assert self.choice in self.choices, "'" + choice + "'" + ' not implemented, please select from ' + str(
            self.choices)

        if choice == "Sharp Edges":
            self.p = 1
            temp = torch.distributions.laplace.Laplace(torch.tensor([0.]),torch.tensor([2]))
            uniform = Uniform(torch.tensor([-3]), torch.tensor([3]))
            uniform2 = Uniform(torch.tensor([-5.0]), torch.tensor([5.0]))
            target = Mixture([uniform,uniform2, temp], torch.tensor([1.,1., 1.]))
            self.target_log_density = lambda samples: target.log_prob(samples.cpu()).to(samples.devices)

        if choice == "Dimension 1":
            self.p = 1
            num_component = 6
            means = torch.tensor([[-0.25], [1.875], [4.125], [6.25], [-5.5],[-8.5]])
            covs = torch.tensor([[[1.]], [[.5]], [[.5]], [[2.]], [[1]],[[1]]])
            comp = torch.ones(num_component)
            mvn_target = MultivariateNormal(means, covs)
            cat = Categorical(comp / torch.sum(comp))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_log_density = lambda samples: mix_target.log_prob(samples.cpu()).to(samples.device)

        if choice == "Blob Dimension 64":
            self.p = 64
            mixtures_target = 4 + 2*torch.randint(0,6,[1])
            L = torch.randn(mixtures_target, self.p, self.p)
            covs_target = L @ L.transpose(-1, -2) + torch.eye(self.p)
            covs_target = 2*covs_target
            means_target = self.p*torch.randn(mixtures_target, self.p)/2
            means_target = means_target
            weights_target = torch.ones(mixtures_target)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_log_density = lambda samples: mix_target.log_prob(samples.cpu()).to(samples.device)

        if choice == "Orbits":
            self.p = 2
            number_planets = 7
            covs_target = 0.04*torch.eye(self.p).unsqueeze(0).repeat(number_planets,1,1)
            means_target = 2.5*torch.view_as_real(torch.pow(torch.exp(torch.tensor([2j * math.pi / number_planets])), torch.arange(0, number_planets)))
            weights_target = torch.ones(number_planets)
            weights_target = weights_target

            mvn_target = MultivariateNormal(means_target, covs_target)
            cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
            mix_target = MixtureSameFamily(cat, mvn_target)
            self.target_log_density = lambda samples: mix_target.log_prob(samples.cpu()).to(samples.device)

    def get_log_density(self):
        return self.target_log_density, self.p

    def target_visual(self):
        return()

