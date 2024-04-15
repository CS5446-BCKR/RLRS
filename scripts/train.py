import typer
from omegaconf import OmegaConf
from path import Path

from rlrs.datasets.food import FoodSimple
from rlrs.envs.food_offline_env import FoodOrderEnv
from rlrs.recommender import Recommender

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    config_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, help="Patht to config file", path_type=Path
    )
):
    cfg = OmegaConf.load(config_path)
    dataset = FoodSimple.from_folder(cfg["input_data"])
    env = FoodOrderEnv(
        dataset,
        state_size=cfg["state_size"],
        user_id=None,
        done_count=cfg["done_count"],
    )
    recommender = Recommender(env, cfg)
    recommender.train()


if __name__ == "__main__":
    app()
