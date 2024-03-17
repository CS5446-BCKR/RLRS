import numpy as np
import torch
from pytest import fixture

from rlrs.nets.state_module import DRRAve

INPUT_DIM = 5
OUTPUT_DIM = 3 * INPUT_DIM
CHECKPOINT = "data/test_data/checkpoints/drrave.pth"


@fixture
def net():
    return DRRAve(INPUT_DIM)


def test_init_drr_ave(net):
    assert net.input_dim == INPUT_DIM
    assert net.output_dim == OUTPUT_DIM


def test_drr_ave(net):

    user = torch.rand(INPUT_DIM)
    items = torch.rand((2, INPUT_DIM))

    inputs = (user, items)
    outputs = net(inputs)
    assert outputs.size() == (OUTPUT_DIM,)


def test_persistent_drr_ave(net):
    net.save(CHECKPOINT)

    new = DRRAve.from_checkpoint(CHECKPOINT)
    old_weight = net.conv.weight.detach().numpy()
    new_weight = new.conv.weight.detach().numpy()
    assert np.allclose(old_weight, new_weight)
    assert new.input_dim == net.input_dim
