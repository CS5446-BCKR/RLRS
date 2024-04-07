import numpy as np
from pytest import fixture

from rlrs.datasets.food import FoodSimple
from rlrs.envs.food_offline_env import FoodOrderEnv

TEST_DATA = "data/test_data/ayampp_lite/"
STATE_SIZE = 3
USER_ID = "zs5V5O4zMPYiKxzO0e2EBy4uq403"
EMPTY_HIS_USER_ID = "iE7KbXeQRrgdHW20acg1NUaKlM73xxx"


def test_loading_offline_env():
    db: FoodSimple = FoodSimple.from_folder(TEST_DATA)
    env: FoodOrderEnv = FoodOrderEnv(
        db,
        state_size=STATE_SIZE,
    )

    assert env is not None
    assert env.database == db
    assert env.num_users == 2
    assert set(env.users.index) == set(
        ["bvkoXpIE2SMiRm6CmOUFJyWZoP62", "zs5V5O4zMPYiKxzO0e2EBy4uq403"]
    )
    assert env.num_items == 8
    assert set(env.items.index) == set(
        [
            "uaaCfJnKgLDbpp5BKjJs",
            "Rp4T9cGa5SptaNJv2nuk",
            "0Xd60T9Jc4hxDjD85u5w",
            "Bab6SE4e8yfncTSfJcbN",
            "sIYfTIFkqU3XYxoRy8G1",
            "uyLAIkFrugXDFupn5cH6",
            "H1jWneJIwy0vU2nTrSMJ",
            "URP9e2B1WdCZkhX5Rl9j",
        ]
    )


def test_loading_specific_env():
    db: FoodSimple = FoodSimple.from_folder(TEST_DATA)
    env: FoodOrderEnv = FoodOrderEnv(db, state_size=STATE_SIZE, user_id=USER_ID)

    assert env.database == db
    assert env.num_users == 1
    assert set(env.users.index) == set([USER_ID])
    assert env.num_items == 8
    assert set(env.items.index) == set(
        [
            "uaaCfJnKgLDbpp5BKjJs",
            "Rp4T9cGa5SptaNJv2nuk",
            "0Xd60T9Jc4hxDjD85u5w",
            "Bab6SE4e8yfncTSfJcbN",
            "sIYfTIFkqU3XYxoRy8G1",
            "uyLAIkFrugXDFupn5cH6",
            "H1jWneJIwy0vU2nTrSMJ",
            "URP9e2B1WdCZkhX5Rl9j",
        ]
    )


@fixture
def env():
    db: FoodSimple = FoodSimple.from_folder(TEST_DATA)
    env: FoodOrderEnv = FoodOrderEnv(db, state_size=STATE_SIZE, user_id=USER_ID)
    return env


@fixture
def neg_env():
    db: FoodSimple = FoodSimple.from_folder(TEST_DATA)
    env: FoodOrderEnv = FoodOrderEnv(
        db,
        state_size=STATE_SIZE,
        user_id=EMPTY_HIS_USER_ID,
    )
    return env


def test_env_reset(env):
    state = env.reset()

    assert state.user_id == USER_ID
    assert state.prev_pos_items == [
        "Rp4T9cGa5SptaNJv2nuk",
        "URP9e2B1WdCZkhX5Rl9j",
        "Bab6SE4e8yfncTSfJcbN",
    ]
    assert env.recommended_items == set(
        ["Rp4T9cGa5SptaNJv2nuk", "URP9e2B1WdCZkhX5Rl9j", "Bab6SE4e8yfncTSfJcbN"]
    )
    assert not state.done
    assert state.reward == 0


def test_empty_historical_positive(neg_env):
    state = neg_env.reset()
    assert state.user_id == EMPTY_HIS_USER_ID
    assert state.prev_pos_items == []
    assert neg_env.recommended_items == set([])
    assert state.done
    assert state.reward == 0


def test_env_step_positive_item(env):
    env.reset()
    state = env.step(["uaaCfJnKgLDbpp5BKjJs"])
    assert np.allclose(state.reward, 1.0)


def test_env_step_negative_item(env):
    env.reset()
    state = env.step(["sIYfTIFkqU3XYxoRy8G1"])
    assert np.allclose(state.reward, -0.1)
