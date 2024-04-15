import pandas as pd
import typer
from path import Path
from sklearn.model_selection import train_test_split

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    input_folder: Path = typer.Argument(
        ..., exists=True, dir_okay=True, help="Input folder", path_type=Path
    ),
    train_ratio: float = typer.Argument(..., help="Training ratio"),
    output_folder: Path = typer.Argument(
        ..., exists=False, dir_okay=True, help="Output folder", path_type=Path
    ),
):

    users = pd.read_csv(input_folder / "users.csv")
    orders = pd.read_csv(input_folder / "orders.csv")
    train_users, test_users = train_test_split(
        users, train_size=train_ratio, random_state=42
    )

    # output
    output_folder.makedirs_p()
    train_dir = output_folder / "train"
    test_dir = output_folder / "test"
    train_dir.makedirs_p()
    test_dir.makedirs_p()

    # copy to train folder
    (input_folder / "foods.csv").copy2(train_dir)
    train_orders = orders[orders.UserID.isin(train_users.UserID)]
    train_users.to_csv(train_dir / "users.csv", index=False)
    train_orders.to_csv(train_dir / "orders.csv", index=False)

    # copy to test folder
    (input_folder / "foods.csv").copy2(test_dir)
    test_users.to_csv(test_dir / "users.csv", index=False)
    test_orders = orders[orders.UserID.isin(test_users.UserID)]
    test_orders.to_csv(test_dir / "orders.csv", index=False)


if __name__ == "__main__":
    app()
