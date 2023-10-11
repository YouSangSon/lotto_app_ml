from domain.entities.entities import AnnuityLottoResult, LottoResult, PredictAnnuityLottoResult, PredictLottoResult
from ports.ports import AnnuityLottoPort, LottoDataPort


class DatabaseAdapter(LottoDataPort, AnnuityLottoPort):
    def __init__(self, session):
        self.session = session

    def get_all_lotto_results(self):
        return self.session.query(LottoResult).all()

    def get_all_annuity_lotto_results(self):
        return self.session.query(AnnuityLottoResult).all()

    def save_predict_lotto_results(self, predicted_lotto_numbers, drw_no, epochs, model_name):
        predicted_lotto_numbers = [int(num) for num in predicted_lotto_numbers]

        if model_name == "lotto":
            predict_result = PredictLottoResult(
                predict_drw_no=int(drw_no),
                drwt_no1=predicted_lotto_numbers[0],
                drwt_no2=predicted_lotto_numbers[1],
                drwt_no3=predicted_lotto_numbers[2],
                drwt_no4=predicted_lotto_numbers[3],
                drwt_no5=predicted_lotto_numbers[4],
                drwt_no6=predicted_lotto_numbers[5],
                predict_epoch=int(epochs),
            )

        elif model_name == "annuity_lotto":
            predict_result = PredictAnnuityLottoResult(
                predict_drw_no=int(drw_no),
                drwt_no1=predicted_lotto_numbers[0],
                drwt_no2=predicted_lotto_numbers[1],
                drwt_no3=predicted_lotto_numbers[2],
                drwt_no4=predicted_lotto_numbers[3],
                drwt_no5=predicted_lotto_numbers[4],
                drwt_no6=predicted_lotto_numbers[5],
                predict_epoch=int(epochs),
            )
        

        # Add the new record to the session
        self.session.add(predict_result)

        # Commit the transaction
        self.session.commit()

        return
