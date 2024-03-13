from torch import nn


def soft_replace_update(src_network: nn.Module, dst_network: nn.Module, tau: float):
    for dst_param, src_param in zip(dst_network.parameters(), src_network.parameters()):
        dst_param.data.copy_(tau * src_param.data + dst_param.data * (1 - tau))


def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).sum() / weight.sum()
