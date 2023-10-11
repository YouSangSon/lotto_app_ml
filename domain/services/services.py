from datetime import datetime

from pytz import timezone
from torch import nn
import numpy as np
import torch

from ports.ports import AnnuityLottoPort, LottoDataPort


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Predictor, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


def extract_win_numbers_from_result(lotto_result):
    return [
        lotto_result.drwt_no1,
        lotto_result.drwt_no2,
        lotto_result.drwt_no3,
        lotto_result.drwt_no4,
        lotto_result.drwt_no5,
        lotto_result.drwt_no6,
    ]


def model_init(input_dim, hidden_dim, output_dim):
    model = Predictor(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


def extract_lotto_drwNo_from_result(lotto_result):
    return [lotto_result.drw_no]


def extract_x_y_train(data):
    X_train = np.array(data[:-1])
    y_train = np.array(data[1:])

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    return X_train, y_train


def get_model_eval(epochs, model, criterion, optimizer, x_train, y_train):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(
                f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, Time: {datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')}"
            )
    model.eval()

    return model


def get_predict_numbers(lotto_data, annuity_lotto_data, lotto_model, annuity_lotto_model):
    with torch.no_grad():
        lotto_last_data = (
            torch.tensor(lotto_data[-1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        annuity_lotto_last_data = (
            torch.tensor(annuity_lotto_data[-1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

        lotto_predictions = lotto_model(lotto_last_data)

        predicted_lotto_numbers = (
                torch.topk(lotto_predictions, 6, dim=-1)[1].squeeze().numpy() + 1
        )

        predicted_lotto_numbers = sorted(predicted_lotto_numbers)
        predicted_lotto_numbers_str = " ".join(map(str, predicted_lotto_numbers))

        annuity_predictions = annuity_lotto_model(annuity_lotto_last_data).view(
            -1, 6, 10
        )  # Reshape output to 6 sets of 10
        predicted_annuity_lotto_numbers = []

        for i in range(6):  # Iterate over each set of 10 predictions
            number_probabilities = torch.softmax(annuity_predictions[0, i], dim=0)
            selected_number = torch.argmax(
                number_probabilities
            ).item()  # Select the number with the highest probability
            predicted_annuity_lotto_numbers.append(selected_number)

        predicted_annuity_lotto_numbers_str = " ".join(
            map(str, predicted_annuity_lotto_numbers)
        )

        return predicted_lotto_numbers, predicted_annuity_lotto_numbers, predicted_lotto_numbers_str, predicted_annuity_lotto_numbers_str


class LottoPredictionService:
    def __init__(self, data_port: LottoDataPort):
        self.data_port = data_port

    def one_hot_encode(self, numbers):
        encoded = np.zeros(45)
        for number in numbers:
            encoded[number - 1] = 1
        return encoded


class AnnuityLottoPredictionService:
    def __init__(self, data_port: AnnuityLottoPort):
        self.data_port = data_port

    def one_hot_encode(self, numbers):
        encoded = np.zeros((6, 10))
        for i, number in enumerate(numbers):
            encoded[i, number] = 1
        return encoded.flatten()
