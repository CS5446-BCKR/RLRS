import random

import pandas as pd
import typer
from path import Path

USER_COLS = ["UserID", "Desc"]
MOVIES_COLS = ["MovieID", "Title"]
RATING_COLS = ["UserID", "MovieID", "Rating", "Timestamp"]

NUM_RATINGS = 20
RATING_RANGE = (0, 5)


def main(
    output_dir: Path = typer.Argument(
        ..., exists=False, dir_okay=True, path_type=Path, help="Path to output folder"
    )
):
    output_dir.makedirs_p()
    random.seed(3443)
    users = []
    user_ID_range = (5, 10)
    for i in range(user_ID_range[0], user_ID_range[1] + 1):
        users.append([i, f"User {i}"])

    users_df = pd.DataFrame(users, columns=USER_COLS)
    users_df.to_csv(output_dir / "users.csv", index=False)

    movies = []
    movie_ID_range = (10, 15)
    for i in range(movie_ID_range[0], movie_ID_range[1] + 1):
        movies.append([i, f"Movie Title {i}"])

    movies_df = pd.DataFrame(movies, columns=MOVIES_COLS)
    movies_df.to_csv(output_dir / "movies.csv", index=False)

    ratings = []
    for i in range(NUM_RATINGS):
        ratings.append(
            [
                random.randint(*user_ID_range),
                random.randint(*movie_ID_range),
                random.randint(*RATING_RANGE),
                i,
            ]
        )
    ratings_df = pd.DataFrame(ratings, columns=RATING_COLS)
    ratings_df.to_csv(output_dir / "ratings.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
