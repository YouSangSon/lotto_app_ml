from ports.ports import NotificationPort
import telegram


class TelegramAdapter(NotificationPort):
    def __init__(self, token):
        self.bot = telegram.Bot(token=token)

    def send_message(self, message, chat_id):
        self.bot.send_message(chat_id=chat_id, text=message)
