from sqlalchemy import Column, Integer, String, Date, Numeric, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Scenario(Base):
    __tablename__ = "scenarios"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    description = Column(String)
    liquidity_shock = Column(Numeric(5, 2), default=0)
    ir_shift = Column(Numeric(5, 2), default=0)
    credit_shock = Column(Numeric(5, 2), default=0)

class BalanceSheet(Base):
    __tablename__ = "balance_sheet"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    item = Column(String(50), nullable=False)
    amount = Column(Numeric(18, 2), nullable=False)
    scenario_id = Column(Integer, ForeignKey('scenarios.id'))

class Cashflow(Base):
    __tablename__ = "cashflows"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    product = Column(String(50), nullable=False)
    counterparty = Column(String(50), nullable=False)
    maturity_date = Column(Date)
    bucket = Column(String(20))
    amount = Column(Numeric(18, 2), nullable=False)
    direction = Column(String(10), nullable=False)
    hqlatype = Column(String(20), nullable=False)
    asf_factor = Column(Numeric(5, 2), default=0)
    rsf_factor = Column(Numeric(5, 2), default=0)
    scenario_id = Column(Integer, ForeignKey('scenarios.id'))

class RWA(Base):
    __tablename__ = "rwa"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    exposure_id = Column(String(50), nullable=False)
    asset_class = Column(String(50), nullable=False)
    approach = Column(String(20), nullable=False)
    amount = Column(Numeric(18, 2), nullable=False)
    risk_weight = Column(Numeric(5, 2), nullable=False)
    rwa_amount = Column(Numeric(18, 2), nullable=False)
    capital_requirement = Column(Numeric(18, 2), nullable=False)
    scenario_id = Column(Integer, ForeignKey('scenarios.id'))

class IRRBB(Base):
    __tablename__ = "irrbb"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    instrument = Column(String(50), nullable=False)
    cashflow = Column(Numeric(18, 2), nullable=False)
    maturity_date = Column(Date, nullable=False)
    tenor_bucket = Column(String(20))
    pv01 = Column(Numeric(10, 6), nullable=False)
    rate_sensitivity = Column(Numeric(10, 6))
    scenario_id = Column(Integer, ForeignKey('scenarios.id'))

class Param(Base):
    __tablename__ = "params"
    key = Column(String(50), primary_key=True)
    value = Column(String(100), nullable=False)
