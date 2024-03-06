from path import Path
from omegaconf import OmegaConf, DictConfig
from rlrs.movie_recommender import MovieRecommender
from rlrs.foodorder_recommender import FoodOrderRecommender
from rlrs.train import RecommenderTrainer
import typer


def load_recommender(config: DictConfig):
    if config["type"] == "movie":
        return MovieRecommender(config)
    elif config["type"] == "food":
        return FoodOrderRecommender(config)
    else:
        raise RuntimeError(f"Invalid recommender: {config['type']}")


def main(
    config_path: Path = typer.Argument(
        ...,
        exists=True, file_okay=True, readable=True,
        help="Path to config file",
        path_type=Path
    )
):
    config = OmegaConf.load(config_path)
    recommender = load_recommender(config)
    trainer = RecommenderTrainer(recommender, config)
    trainer.train()


if __name__ == "__main__":
    typer.run(main)
