import numpy as np
import torch
from pytest import fixture

from rlrs.nets.critic import Critic, CriticNetwork

ACTION_DIM = 6
STATE_DIM = 8
EMBED_DIM = 10
HIDDEN_DIM = 12
TAU = 0.001
LR = 0.001
STEP_SIZE = 200


@fixture
def critic_net():
    return CriticNetwork(
        input_action_dim=ACTION_DIM,
        input_state_dim=STATE_DIM,
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    )


@fixture
def model_input():
    action = torch.randn(ACTION_DIM)
    state = torch.randn(STATE_DIM)
    return (action, state)


def test_critic_network(critic_net):

    assert critic_net.input_action_dim == ACTION_DIM
    assert critic_net.input_state_dim == STATE_DIM
    assert critic_net.embedding_dim == EMBED_DIM
    assert critic_net.hidden_dim == HIDDEN_DIM


def test_forward_critic_network(critic_net, model_input):
    outputs = critic_net(model_input)
    assert outputs.size() == (1,)
    assert isinstance(outputs.item(), float)


@fixture
def net():
    return Critic(
        input_action_dim=ACTION_DIM,
        input_state_dim=STATE_DIM,
        embedding_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        tau=TAU,
        lr=LR,
        step_size=STEP_SIZE,
    )


def test_init_critic(net):
    assert net.input_state_dim == STATE_DIM
    assert net.input_action_dim == ACTION_DIM
    assert net.embedding_dim == EMBED_DIM
    assert net.hidden_dim == HIDDEN_DIM
    assert net.tau == TAU
    assert net.lr == LR
    assert net.step_size == STEP_SIZE


def test_init_weight_critic(net):
    net.initialize()
    target = net.target
    online = net.online_network
    target_param_dict = target.state_dict()
    for name, param in online.state_dict().items():
        assert name in target_param_dict
        target_param = target_param_dict[name]
        assert np.allclose(param.numpy(), target_param.numpy())


def test_forward_critic(net, model_input):
    net.initialize()

    online_outputs = net.forward(model_input).detach().numpy()
    target_outputs = net.target_forward(model_input).detach().numpy()

    assert np.allclose(online_outputs, target_outputs)


def test_fit_critic(net, model_input): ...


def test_dq_da_critic(net, model_input): ...


def test_load_from_checkpoint(net): ...


def test_load_from_config(): ...
