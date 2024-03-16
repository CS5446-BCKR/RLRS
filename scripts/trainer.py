import typer
from omegaconf import DictConfig, OmegaConf
from path import Path

from rlrs.datasets.movielens import MovieLens
from rlrs.envs.offline_env import OfflineEnv
from rlrs.foodorder_recommender import FoodOrderRecommender
from rlrs.movie_recommender import MovieRecommender

# from rlrs.train import RecommenderTrainer


def load_recommender(config: DictConfig):
    if config["type"] == "movie":
        db = MovieLens.from_folder()
        env = OfflineEnv(db, state_size=config["state_size"])
        return MovieRecommender(env, config)
    elif config["type"] == "food":
        return FoodOrderRecommender(config)
    else:
        raise RuntimeError(f"Invalid recommender: {config['type']}")


def main(
    config_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        readable=True,
        help="Path to config file",
        path_type=Path,
    )
):
    config = OmegaConf.load(config_path)
    recommender = load_recommender(config)
    recommender.train()


if __name__ == "__main__":
    typer.run(main)
