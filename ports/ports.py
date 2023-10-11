from abc import ABC, abstractmethod


class LottoDataPort(ABC):
    @abstractmethod
    def __init__(self, session):
        pass


class AnnuityLottoPort(ABC):
    @abstractmethod
    def __init__(self, session):
        pass


class NotificationPort(ABC):
    @abstractmethod
    def send_message(self, message, chat_id):
        pass
