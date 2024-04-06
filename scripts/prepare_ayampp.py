import math
from collections import defaultdict

import pandas as pd
import typer
from path import Path

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    input_folder: Path = typer.Argument(
        ..., exists=True, dir_okay=True, help="Path to ayampp data", path_type=Path
    ),
    output_folder: Path = typer.Argument(
        ..., exists=False, dir_okay=True, help="Path to output dir", path_type=Path
    ),
):
    orders = pd.read_csv(input_folder / "orders.csv")
    num_orders = defaultdict(int)

    # create order dataframe
    order_pairs = []
    for _, order in orders.iterrows():
        userid = order["customer_id"]
        timestamp = order["created_at"]
        for item_col in range(1, 65):
            item = order[f"item{item_col}"]
            if isinstance(item, float) and math.isnan(item):
                continue
            order_pairs.append([userid, item, timestamp])
            num_orders[userid] += 1

    output_folder.makedirs_p()
    orders_df = pd.DataFrame(order_pairs, columns=["UserID", "ItemID", "Timestamp"])
    orders_df.to_csv(output_folder / "orders.csv", index=False)

    # create items dataframe
    items = pd.read_csv(input_folder / "menu_items.csv")
    items = items[["id", "name", "description", "price"]]
    items = items.rename(columns={"id": "ItemID"})
    items.to_csv(output_folder / "foods.csv", index=False)

    # create users dataframe
    users = pd.DataFrame(list(num_orders.items()), columns=["UserID", "Num_Item"])
    users.to_csv(output_folder / "users.csv", index=False)


if __name__ == "__main__":
    app()
