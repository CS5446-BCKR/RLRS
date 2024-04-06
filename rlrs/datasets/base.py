from abc import ABC, abstractmethod, abstractproperty


class Dataset(ABC):
    @abstractmethod
    def get_users_by_history(self, threshold: int): ...

    @abstractmethod
    def get_rating(self, user, item): ...

    @abstractmethod
    def get_user_history_length(self, user): ...

    @abstractmethod
    def get_positive_items(self, user, **kwargs): ...

    @abstractproperty
    def num_items(self): ...

    @abstractproperty
    def num_users(self): ...

    @abstractproperty
    def items(self): ...

    @abstractmethod
    def get_rating_matrix(self, **kwargs): ...
