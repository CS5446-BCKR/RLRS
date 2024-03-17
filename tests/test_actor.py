import numpy as np
import torch
from omegaconf import OmegaConf
from pytest import fixture

from rlrs.nets.actor import Actor, ActorModel

INPUT_DIM = 5
HIDDEN_DIM = 10
OUTPUT_DIM = 8
TAU = 0.3
LR = 0.001
STEP_SIZE = 200
SAVE_PATH = "data/test_data/checkpoints/actor.pth"

DICT_CONFIG = OmegaConf.create(
    {
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "tau": TAU,
        "lr": LR,
        "step_size": STEP_SIZE,
    }
)


def test_init_actor():
    actor = Actor(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        tau=TAU,
        lr=LR,
        step_size=STEP_SIZE,
    )

    assert isinstance(actor.target, ActorModel)
    assert isinstance(actor.online_network, ActorModel)
    assert actor.input_dim == INPUT_DIM
    assert actor.hidden_dim == HIDDEN_DIM
    assert actor.output_dim == OUTPUT_DIM
    assert actor.tau == TAU
    assert actor.lr == LR
    assert actor.step_size == STEP_SIZE


@fixture
def net():
    return Actor(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        tau=TAU,
        lr=LR,
        step_size=STEP_SIZE,
    )


def test_init_weight_actor(net):
    net.initialize()
    target = net.target
    online = net.online_network
    target_param_dict = target.state_dict()
    for name, param in online.state_dict().items():
        assert name in target_param_dict
        target_param = target_param_dict[name]
        assert np.allclose(param.numpy(), target_param.numpy())


def test_forward_actor(net):
    net.initialize()
    inputs = torch.randn(INPUT_DIM)

    online_outputs = net.forward(inputs).detach().numpy()
    target_outputs = net.target_forward(inputs).detach().numpy()

    assert np.allclose(online_outputs, target_outputs)


def test_fit_actor(net):
    net.initialize()
    inputs = torch.randn(INPUT_DIM)
    grads = torch.randn(OUTPUT_DIM)
    net.fit(inputs, grads)


def test_init_actor_from_checkpoint(net):
    net.initialize()
    net.save(SAVE_PATH)
    new = Actor.from_checkpoint(SAVE_PATH)
    new_param_dict = new.state_dict()
    for name, param in net.state_dict().items():
        assert name in new_param_dict
        assert np.allclose(param, new_param_dict[name])


def test_init_actor_from_config():
    actor = Actor.from_config(DICT_CONFIG)
    assert isinstance(actor.target, ActorModel)
    assert isinstance(actor.online_network, ActorModel)
    assert actor.input_dim == INPUT_DIM
    assert actor.hidden_dim == HIDDEN_DIM
    assert actor.output_dim == OUTPUT_DIM
    assert actor.tau == TAU
    assert actor.lr == LR
    assert actor.step_size == STEP_SIZE
