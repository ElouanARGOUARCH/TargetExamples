import torch
from torch import nn
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily, Uniform
import math



class VariationalInferenceTarget(nn.Module):
    def __init__(self):
        super().__init__()

    def log_prob(self):
        raise NotImplementedError

class Orbits(VariationalInferenceTarget):
    def __init__(self):
        super().__init__()
        self.p = 2
        number_planets = 7
        covs_target = 0.04*torch.eye(self.p).unsqueeze(0).repeat(number_planets,1,1)
        means_target = 2.5*torch.view_as_real(torch.pow(torch.exp(torch.tensor([2j * math.pi / number_planets])), torch.arange(0, number_planets)))
        weights_target = torch.ones(number_planets)
        weights_target = weights_target

        mvn_target = MultivariateNormal(means_target, covs_target)
        cat = Categorical(torch.exp(weights_target) / torch.sum(torch.exp(weights_target)))
        self.mix_target = MixtureSameFamily(cat, mvn_target)

    def log_prob(self, samples):
        return self.mix_target.log_prob(samples.cpu()).to(samples.device)

class Banana(VariationalInferenceTarget):
    def __init__(self):
        super().__init__()
        var = 2
        self.p = 50
        self.even = torch.arange(0, self.p, 2)
        self.odd = torch.arange(1, self.p, 2)
        self.mvn = torch.distributions.MultivariateNormal(torch.zeros(self.p), var * torch.eye(self.p))

    def inv_transform(self, z):
        x = z.clone()
        x[..., self.odd] -= x[..., self.even] ** 2

        return x

    def log_prob(self, samples):
        return self.mvn.log_prob(self.inv_transform(samples))

class Dimension1(VariationalInferenceTarget):
    def __init__(self):
        super().__init__()
        self.p = 1
        temp = torch.distributions.laplace.Laplace(torch.tensor([0.]),torch.tensor([2]))
        uniform = Uniform(torch.tensor([-3]), torch.tensor([3]))
        uniform2 = Uniform(torch.tensor([-5.0]), torch.tensor([5.0]))
        self.mvn = Mixture([uniform,uniform2, temp], torch.tensor([1.,1., 1.]))

    def log_prob(self, samples):
        return self.mvn.log_prob(samples.cpu()).to(samples.devices)

class BlobDimension64(VariationalInferenceTarget):
    def __init__(self):
        super().__init__()
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
        self.mvn = MixtureSameFamily(cat, mvn_target)

    def log_prob(self, samples):
        return self.mvn.log_prob(samples.cpu()).to(samples.device)
