import json

import h5py
import typer
from path import Path

from rlrs.datasets.base import Dataset
from rlrs.datasets.food import FoodSimple
from rlrs.datasets.movielens import MovieLens
from rlrs.embeddings.mf import SVD

app = typer.Typer(pretty_exceptions_show_locals=False)


def dataset_loader(name: str, dir_path: Path) -> Dataset:
    if name == "movie":
        return MovieLens.from_folder(dir_path)
    elif name == "food":
        return FoodSimple.from_folder(dir_path)
    raise RuntimeError(f"invalid dataset name: {name}")


@app.command()
def main(
    dataset_name: str = typer.Argument(..., help="movie or food"),
    folder_path: Path = typer.Argument(
        ..., exists=True, dir_okay=True, help="Path to dataset folder", path_type=Path
    ),
    dim: int = typer.Argument(..., help="Embedding size"),
    output_folder: Path = typer.Argument(
        ..., exists=False, dir_okay=True, help="Path to output fodler", path_type=Path
    ),
):
    dataset = dataset_loader(dataset_name, folder_path)
    config = {"rating_threshold": 3}
    ratings = dataset.get_rating_matrix(**config)
    algo = SVD(dim)
    algo.fit(ratings)
    user_embeds = algo.user_matrix
    item_embeds = algo.item_matrix

    # save
    output_folder.makedirs_p()
    json.dump(
        {
            "dim": dim,
            "method": "SVD",
            "dataset_name": dataset_name,
            "input_folder_path": folder_path,
        },
        open(output_folder / "config.json", "w"),
    )
    with h5py.File(output_folder / "embeddings.h5", "w") as h5file:
        h5file.create_dataset("user_embedding", data=user_embeds)
        h5file.create_dataset("item_embedding", data=item_embeds)
        h5file.create_dataset("user_index", data=dataset.users.index.values)
        h5file.create_dataset("item_index", data=dataset.items.index.values)


if __name__ == "__main__":
    app()
