import asyncio
import random
import string

import telegram

from adapters.db_adapter import DatabaseAdapter
from adapters.telegram_adapter import TelegramAdapter

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import torch
import torch.nn as nn

import json
import os

from domain.services.services import (
    LottoPredictionService,
    AnnuityLottoPredictionService,
    extract_lotto_drwNo_from_result,
    extract_win_numbers_from_result,
    extract_x_y_train,
    get_model_eval,
    get_predict_numbers,
    model_init,
)


class ConfigFile:
    database_url = string
    telegram_token = string
    telegram_chat_id = string
    random_start = int
    random_end = int
    hidden_dim = int


def config_init():
    new_config_file = ConfigFile()

    with open("config.json", "r") as f:
        config = json.load(f)
        new_config_file.database_url = config["database_url"]
        new_config_file.telegram_token = config["telegram_token"]
        new_config_file.telegram_chat_id = config["telegram_chat_id"]
        new_config_file.random_start = config["random_start"]
        new_config_file.random_end = config["random_end"]
        new_config_file.hidden_dim = config["hidden_dim"]

    return new_config_file


# def create_tables():
#     Base.metadata.create_all(bind=engine)


# create_tables()

def main():
    # os setup
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    device = torch.device("cpu")  # Default device

    # config init
    config_file = config_init()

    # telegram setup
    # telegram_service = TelegramAdapter(
    #     token=config_file.telegram_token
    # )
    bot = telegram.Bot(token=config_file.telegram_token)

    # db setup
    engine = create_engine(str(config_file.database_url), echo=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db_session = SessionLocal()
    lotto_db_session = DatabaseAdapter(db_session)

    # service init
    lotto_prediction_service = LottoPredictionService(lotto_db_session)
    annuity_lotto_prediction_service = AnnuityLottoPredictionService(lotto_db_session)

    all_lotto_results = lotto_db_session.get_all_lotto_results()
    all_annuity_lotto_results = lotto_db_session.get_all_annuity_lotto_results()

    lotto_numbers_list = [
        extract_win_numbers_from_result(result) for result in all_lotto_results
    ]
    annuity_lotto_numbers_list = [
        extract_win_numbers_from_result(result) for result in all_annuity_lotto_results
    ]

    lotto_drwNo_list = sorted(
        [extract_lotto_drwNo_from_result(result) for result in all_lotto_results]
    )

    annuity_lotto_drwNo_list = sorted(
        [
            extract_lotto_drwNo_from_result(result)
            for result in all_annuity_lotto_results
        ]
    )

    # ml init
    lotto_model, lotto_optimizer = model_init(45, config_file.hidden_dim, 45)
    annuity_lotto_model, annuity_lotto_optimizer = model_init(
        60, config_file.hidden_dim, 60
    )
    criterion = nn.BCEWithLogitsLoss()

    lotto_data = [
        lotto_prediction_service.one_hot_encode(numbers) for numbers in lotto_numbers_list
    ]
    annuity_lotto_data = [
        annuity_lotto_prediction_service.one_hot_encode(numbers) for numbers in annuity_lotto_numbers_list
    ]

    lotto_X_train, lotto_y_train = extract_x_y_train(lotto_data)
    annuity_lotto_X_train, annuity_lotto_y_train = extract_x_y_train(annuity_lotto_data)

    lotto_epochs = random.randint(config_file.random_start, config_file.random_end)
    annuity_epochs = random.randint(config_file.random_start, config_file.random_end)
    # lotto_epochs = 1000
    # annuity_epochs = 1000

    lotto_model = get_model_eval(
        lotto_epochs,
        lotto_model,
        criterion,
        lotto_optimizer,
        lotto_X_train,
        lotto_y_train,
    )
    annuity_lotto_model = get_model_eval(
        annuity_epochs,
        annuity_lotto_model,
        criterion,
        annuity_lotto_optimizer,
        annuity_lotto_X_train,
        annuity_lotto_y_train,
    )

    (
        predicted_lotto_numbers, predicted_annuity_lotto_numbers,
        predicted_lotto_numbers_str,
        predicted_annuity_lotto_numbers_str,
    ) = get_predict_numbers(
        lotto_data, annuity_lotto_data, lotto_model, annuity_lotto_model
    )

    asyncio.run(bot.send_message(
        chat_id=str(config_file.telegram_chat_id),
        text="예상 로또 번호 : " + predicted_lotto_numbers_str + "\n예상 연금 로또 번호 : " +
             predicted_annuity_lotto_numbers_str
    ))

    lotto_predict_drw_no = len(lotto_drwNo_list) + 1
    annuity_lotto_predict_drw_no = len(annuity_lotto_drwNo_list) + 1

    lotto_db_session.save_predict_lotto_results(
        predicted_lotto_numbers=predicted_lotto_numbers,
        drw_no=lotto_predict_drw_no,
        epochs=lotto_epochs,
        model_name="lotto",
    )
    lotto_db_session.save_predict_lotto_results(
        predicted_lotto_numbers=predicted_annuity_lotto_numbers,
        drw_no=annuity_lotto_predict_drw_no,
        epochs=annuity_epochs,
        model_name="annuity_lotto",
    )

    lotto_db_session.session.close()


if __name__ == "__main__":
    main()
