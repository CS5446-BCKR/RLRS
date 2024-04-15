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
        assert len(groundtruths[user]) > 0
    return groundtruths

def get_pretty_names(item_id_list, dataset: FoodSimple):
    return dataset.foods[dataset.foods.index.isin(item_id_list)].name.tolist()

def evaluate(recommender: Recommender, dataset: FoodSimple,  user, gts, top_k, verbal=False):
    """
    1-step evaluation
    """
    recommender.set_users([user])
    recommender.reset()

    if verbal:
        user_hist_len = recommender.env.db.get_user_history_length(user)
        history_items = recommender.env.db.get_positive_items(user)
        history_item_names = get_pretty_names(history_items,dataset)
        gts_item_name = get_pretty_names(gts,dataset)
        print(f"user_id : {user}, user_history_length:{user_hist_len}")
        print(f"history items : \n {history_item_names}")
        print(f"Groundtruths: \n {gts_item_name}")

    recommended_items = recommender.recommend()

    true_pos = set(recommended_items) & set(gts)
    precision = len(true_pos) / top_k
    r1 = int(len(true_pos) > 0)
    if verbal:
        rec_item_names = get_pretty_names(recommended_items.tolist(),dataset)
        tp_names = get_pretty_names(list(true_pos),dataset)

        print(f"recommended items : {rec_item_names}")
        print(f"true positives: {tp_names}")
        print(f"Prec@{top_k} {user}: {precision}")
        print(f"Recall-1@{top_k} {user}: {r1}")
        print("=======")
    return precision, r1


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
    r1_at_k = []


    groundtruths = prepare_groundtruths(dataset, state_size)

    for user, gts in groundtruths.items():
        precision, r1 = evaluate(recommender, dataset , user, gts, top_k=topk, verbal=verbose)
        precs.append(precision)
        r1_at_k.append(r1)

    print(f"Avg Precision@{topk}: {np.mean(precs)}")
    print(f"Avg Recall-1@{topk}: {np.mean(r1_at_k)}")


if __name__ == "__main__":
    typer.run(main)
