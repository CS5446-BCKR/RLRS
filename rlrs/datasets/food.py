from pathlib import Path

import numpy as np
import pandas as pd

from .base import Dataset

USER_IDX_COL = "UserID"  # user_id
ITEM_IDX_COL = "ItemID"  # item_id


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
        self.UserItems: pd.DataFrame = (
            UserItems  # equivalent of ratings table in movielens
        )

        self.idToMenuItem = self.menuItems.set_index("id").to_dict("index")

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


class FoodSimple(Dataset):
    def __init__(self, foods: pd.DataFrame, users: pd.DataFrame, orders: pd.DataFrame):
        self.users_ = users.set_index(USER_IDX_COL)
        self.foods = foods.set_index(ITEM_IDX_COL)
        self.orders = orders
        self.freq = (
            self.orders[[USER_IDX_COL, ITEM_IDX_COL]].groupby(USER_IDX_COL).agg(len)
        )

    @property
    def num_items(self):
        return len(self.foods)

    @property
    def num_users(self):
        return len(self.users)

    @property
    def items(self):
        return self.foods

    @property
    def users(self):
        return self.users_

    @classmethod
    def from_folder(cls, src: Path):
        src = Path(src)
        return cls(
            pd.read_csv(src / "foods.csv"),
            pd.read_csv(src / "users.csv"),
            pd.read_csv(src / "orders.csv"),
        )

    def get_user_history_length(self, user):
        return len(self.orders[self.orders[USER_IDX_COL] == user])

    def get_users_by_history(self, threshold):
        return self.freq.index[self.freq[ITEM_IDX_COL] >= threshold].tolist()

    def get_rating(self, user, item):
        return len(
            self.orders[
                (self.orders[USER_IDX_COL] == user)
                & (self.orders[ITEM_IDX_COL] == item)
            ]
        )
        res = self.orders.query(f"UserID == '{user}' and ItemID == {item}")
        return len(res)

    def get_positive_items(self, user, **kwargs):
        res = self.orders.query(f"UserID == '{user}'")[ITEM_IDX_COL]
        return res.tolist()

    def get_rating_matrix(self, **kwargs):
        """
        Rating = how many time the users order that item
        normalized by their total number of item ordered.
        """
        dataframe = pd.DataFrame(
            np.zeros((self.num_users, self.num_items)),
            index=self.users.index,
            columns=self.items.index,
        )
        freq = self.orders.groupby([USER_IDX_COL, ITEM_IDX_COL]).agg(len)
        for index, row in freq.itertuples(index=True):
            dataframe.loc[index] += row
        dataframe = dataframe.values
        dataframe /= dataframe.sum(axis=1, keepdims=True)
        return pd.DataFrame(
            dataframe,
            index=self.users.index,
            columns=self.items.index,
        )
