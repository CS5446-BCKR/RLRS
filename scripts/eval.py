import numpy as np
import typer
from omegaconf import OmegaConf
from path import Path

from rlrs.datasets.food import FoodSimple
from rlrs.envs.food_offline_env import FoodOrderEnv
from rlrs.recommender import Recommender

AYAMPP_LITE_CFG = "configs/ayampp_small_base_infer.yaml"


app = typer.Typer(pretty_exceptions_show_locals=False)


def prepare_groundtruths(dataset: FoodSimple, state_size: int):
    """
    Find all users with histories > state_size
    All items whose indexes > state_size are considered
    as groundtruths for evaluation
    """
    users_with_history = dataset.get_users_by_history(state_size + 1)
    groundtruths = {}
    for user in users_with_history:
        groundtruths[user] = dataset.get_positive_items(user)[state_size:]
    return groundtruths


def evaluate(recommender: Recommender, user, gts, top_k, verbal=False):
    """
    1-step evaluation
    """
    recommender.set_users([user])
    recommender.reset()

    if verbal:
        user_hist_len = recommender.env.db.get_user_history_length(user)
        history_items = recommender.env.db.get_positive_items(user)
        print(f"user_id : {user}, user_history_length:{user_hist_len}")
        """TODO: missing get_items_names equivalent"""
        print(f"history items : \n {history_items}")
        print(f"Groundtruths: \n {gts}")

    recommended_items = recommender.recommend()

    true_pos = set(recommended_items) & set(gts)
    precision = len(true_pos) / top_k
    if verbal:
        print(f"recommended items ids : {recommended_items}")
        print(f"true positives: {true_pos}")
        print(f"Prec@{top_k} {user}: {precision}")
        print("=======")
    return precision


@app.command()
def main(
    config_path: Path = typer.Argument(
        ..., file_okay=True, exists=True, help="Path to the config file", path_type=Path
    ),
    verbose: bool = typer.Option(False, help="verbose mode"),
):
    cfg = OmegaConf.load(config_path)
    state_size = cfg["state_size"]
    dataset = FoodSimple.from_folder(cfg["input_data"])
    env = FoodOrderEnv(
        dataset,
        state_size=state_size,
        user_id=None,
        done_count=cfg["done_count"],
    )
    recommender = Recommender(env, cfg)
    topk = cfg["topk"]
    precs = []

    groundtruths = prepare_groundtruths(dataset, state_size)

    for user, gts in groundtruths.items():
        precision = evaluate(recommender, user, gts, top_k=topk, verbal=verbose)
        precs.append(precision)

    print(f"Avg Precision@{topk}: {np.mean(precs)}")


if __name__ == "__main__":
    typer.run(main)
