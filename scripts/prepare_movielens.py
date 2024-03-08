from typing import List

import pandas as pd
import typer
from path import Path

COL_NAMES = {
    "movies.dat": ["MovieID", "Title", "Genres"],
    "ratings.dat": ["UserID", "MovieID", "Rating", "Timestamp"],
    "users.dat": ["UserID", "Gender", "Age", "Occupation", "ZipCode"],
}


def prepare_file(src_file: Path, col_names: List[str], dst_file: Path):
    df = pd.read_table(
        src_file,
        sep="::",
        header=None,
        names=col_names,
        encoding="latin-1",
        engine="python",
    )
    df.to_csv(dst_file, index=False)


def main(
    source_dir: Path = typer.Argument(
        ..., help="Path to source folder", dir_okay=True, exists=True, path_type=Path
    ),
    dst_dir: Path = typer.Argument(
        ...,
        help="Path to destination folder",
        dir_okay=True,
        exists=False,
        path_type=Path,
    ),
):
    """
    Prepare MovieLens data, convert custom data format to CSV.
    """
    if not dst_dir.exists():
        dst_dir.makedirs_p()
    for src_name, col_names in COL_NAMES.items():
        dst = dst_dir / f"{Path(src_name).stem}.csv"
        prepare_file(source_dir / src_name, col_names, dst)
    print("Done !!")


if __name__ == "__main__":
    typer.run(main)
