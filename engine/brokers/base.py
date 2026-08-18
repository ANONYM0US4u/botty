from abc import ABC, abstractmethod


class BrokerAdapter(ABC):
    @abstractmethod
    def place_order(self, order: dict) -> dict: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_positions(self) -> list[dict]: ...

    @abstractmethod
    def get_orders(self) -> list[dict]: ...

    @abstractmethod
    def get_balance(self) -> float: ...