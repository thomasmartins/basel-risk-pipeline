"""Pydantic row models + enums for the ALM warehouse.

The pydantic models mirror the warehouse schema. They are the single
source of truth for column names and dtypes used by ingestion, queries,
the risk engine, and (eventually) dbt source contracts.
"""

from __future__ import annotations

from datetime import date as _date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class HQLAType(StrEnum):
    LEVEL1 = "Level1"
    LEVEL2A = "Level2A"
    LEVEL2B = "Level2B"
    NONE = "None"


class Direction(StrEnum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"


class Approach(StrEnum):
    STD = "STD"
    IRB = "IRB"


class AssetClass(StrEnum):
    MORTGAGE = "mortgage"
    CORPORATE = "corporate"
    SOVEREIGN = "sovereign"
    RETAIL = "retail"


class Product(StrEnum):
    LOAN = "loan"
    DEPOSIT = "deposit"
    BOND = "bond"


class Counterparty(StrEnum):
    RETAIL = "retail"
    WHOLESALE = "wholesale"


class BalanceSheetItem(StrEnum):
    CET1 = "CET1"
    TIER1 = "Tier1"
    TOTAL_CAPITAL = "Total Capital"
    TOTAL_ASSETS = "Total Assets"
    TOTAL_LIABILITIES = "Total Liabilities"


class TenorBucket(StrEnum):
    Y0_1 = "0-1y"
    Y1_3 = "1-3y"
    Y3_5 = "3-5y"
    Y5_10 = "5-10y"
    Y10_PLUS = "10y+"


_BASE = ConfigDict(frozen=True, extra="forbid")


class Scenario(BaseModel):
    model_config = _BASE
    id: int
    name: str
    description: str | None = None
    liquidity_shock: Decimal = Field(default=Decimal("0"))
    ir_shift: Decimal = Field(default=Decimal("0"))
    credit_shock: Decimal = Field(default=Decimal("0"))


class BalanceSheetRow(BaseModel):
    model_config = _BASE
    id: int
    date: _date
    item: BalanceSheetItem
    amount: Decimal
    scenario_id: int | None = None


class CashflowRow(BaseModel):
    model_config = _BASE
    id: int
    date: _date
    product: Product
    counterparty: Counterparty
    maturity_date: _date | None
    bucket: str | None
    amount: Decimal
    direction: Direction
    hqlatype: HQLAType
    asf_factor: Decimal = Field(default=Decimal("0"))
    rsf_factor: Decimal = Field(default=Decimal("0"))
    scenario_id: int | None = None


class RWARow(BaseModel):
    model_config = _BASE
    id: int
    date: _date
    exposure_id: str
    asset_class: AssetClass
    approach: Approach
    amount: Decimal
    risk_weight: Decimal
    rwa_amount: Decimal
    capital_requirement: Decimal
    scenario_id: int | None = None


class IRRBBRow(BaseModel):
    model_config = _BASE
    id: int
    date: _date
    instrument: str
    cashflow: Decimal
    maturity_date: _date
    tenor_bucket: TenorBucket | None
    pv01: Decimal
    rate_sensitivity: Decimal | None = None
    scenario_id: int | None = None


class ParamRow(BaseModel):
    model_config = _BASE
    key: str
    value: str
