from pathlib import Path

import pandas as pd


class Food:

    def __init__(
        self,
        menuItems: pd.DataFrame,
        modGroups: pd.DataFrame,
        modItems: pd.DataFrame,
        orders: pd.DataFrame,
        orderItems: pd.DataFrame,
        UserItems: pd.DataFrame,
    ):
        self.menuItems: pd.DataFrame = menuItems
        self.modGroups: pd.DataFrame = modGroups
        self.modItems: pd.DataFrame = modItems
        self.orders: pd.DataFrame = orders
        self.orderItems: pd.DataFrame = orderItems
        self.UserItems: pd.DataFrame = UserItems # equivalent of ratings table in movielens

        self.idToMenuItem = self.menuItems.set_index("id").to_dict("index")
        self.idToModGroup = self.modGroups.set_index("id").to_dict("index")
        self.idToModItem = self.modItems.set_index("id").to_dict("index")
        self.idToOrder = self.orders.set_index("order_id").to_dict("index")

    @classmethod
    def from_folder(cls, src: Path):
        return cls(
            pd.read_csv(src / "menu_items.csv"),
            pd.read_csv(src / "mod_groups.csv"),
            pd.read_csv(src / "mod_items.csv"),
            pd.read_csv(src / "orders.csv", dtype=str, keep_default_na=False),
            pd.read_csv(src / "order_items.csv"),
            pd.read_csv(src / "user_item_frequency_table.csv"),
        )


# ayampp = Food.from_folder(Path(__file__).parent.resolve() / "../../data/ayampp")
# print(ayampp.idToOrder["es6LCvJJwdS8kSvEZpE6"])
