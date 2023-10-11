from sqlalchemy import BigInteger, Integer, Date, Column, String, Numeric, DateTime
import sqlalchemy.orm

Base = sqlalchemy.orm.declarative_base()


class LottoResult(Base):
    __tablename__ = "lotto_results"

    id = Column(BigInteger, primary_key=True)
    tot_sellamnt = Column(BigInteger)
    bnus_no = Column(Integer)
    drw_no = Column(Integer)
    drw_no_date = Column(Date)
    drwt_no1 = Column(Integer)
    drwt_no2 = Column(Integer)
    drwt_no3 = Column(Integer)
    drwt_no4 = Column(Integer)
    drwt_no5 = Column(Integer)
    drwt_no6 = Column(Integer)
    first_accumamnt = Column(BigInteger)
    first_przwner_co = Column(Integer)
    first_winamnt = Column(BigInteger)
    return_value = Column(String)


class PredictLottoResult(Base):
    __tablename__ = "predict_lotto_results"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    predict_drw_no = Column(Integer)
    drwt_no1 = Column(Integer)
    drwt_no2 = Column(Integer)
    drwt_no3 = Column(Integer)
    drwt_no4 = Column(Integer)
    drwt_no5 = Column(Integer)
    drwt_no6 = Column(Integer)
    predict_per = Column(Numeric(5, 5))
    predict_epoch = Column(BigInteger)


class AnnuityLottoResult(Base):
    __tablename__ = "annuity_lotto_results"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    drw_no = Column(Integer, nullable=False)
    drw_no_date = Column(DateTime, nullable=False)
    group = Column(Integer, nullable=False)
    drwt_no1 = Column(Integer, nullable=False)
    drwt_no2 = Column(Integer, nullable=False)
    drwt_no3 = Column(Integer, nullable=False)
    drwt_no4 = Column(Integer, nullable=False)
    drwt_no5 = Column(Integer, nullable=False)
    drwt_no6 = Column(Integer, nullable=False)
    bonus_no1 = Column(Integer, nullable=False)
    bonus_no2 = Column(Integer, nullable=False)
    bonus_no3 = Column(Integer, nullable=False)
    bonus_no4 = Column(Integer, nullable=False)
    bonus_no5 = Column(Integer, nullable=False)
    bonus_no6 = Column(Integer, nullable=False)


class PredictAnnuityLottoResult(Base):
    __tablename__ = "predict_annuity_lotto_results"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    predict_drw_no = Column(Integer)
    drwt_no1 = Column(Integer)
    drwt_no2 = Column(Integer)
    drwt_no3 = Column(Integer)
    drwt_no4 = Column(Integer)
    drwt_no5 = Column(Integer)
    drwt_no6 = Column(Integer)
    predict_per = Column(Numeric(5, 5))
    predict_epoch = Column(BigInteger)
