from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class TransactionBase(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    transaction_date: date = Field(..., alias="Date")
    account_type: str = Field(..., alias="Account Type")
    transaction_amount: float = Field(..., alias="Transaction Amount")
    cash_flow: float = Field(..., alias="Cash Flow")
    net_income: float = Field(..., alias="Net Income")
    revenue: float | None = Field(default=None, alias="Revenue")
    expenditure: float = Field(..., alias="Expenditure")
    profit_margin: float = Field(..., alias="Profit Margin")
    debt_to_equity_ratio: float = Field(..., alias="Debt-to-Equity Ratio")
    operating_expenses: float = Field(..., alias="Operating Expenses")
    gross_profit: float = Field(..., alias="Gross Profit")
    transaction_volume: float = Field(..., alias="Transaction Volume")
    processing_time_seconds: float = Field(
        ..., alias="Processing Time (seconds)"
    )
    accuracy_score: float = Field(..., alias="Accuracy Score")
    missing_data_indicator: bool = Field(
        ..., alias="Missing Data Indicator"
    )
    normalized_transaction_amount: float = Field(
        ..., alias="Normalized Transaction Amount"
    )


class DSO1PredictRequest(TransactionBase):
    revenue: float = Field(..., alias="Revenue")


class DSO2PredictRequest(TransactionBase):
    revenue: float | None = Field(default=None, alias="Revenue")


class DSO3PredictRequest(TransactionBase):
    revenue: float = Field(..., alias="Revenue")


class DSO1PredictionResponse(BaseModel):
    model: str
    predicted_label: int
    predicted_class: str
    probability_success: float
    probability_failure: float
    confidence: float


class DSO2PredictionResponse(BaseModel):
    model: str
    predicted_value: float
    feature_count: int


class DSO3PredictionResponse(BaseModel):
    model: str
    cluster_id: int
    is_anomaly: bool
    anomaly_score: float
    pca_coordinates: list[float]
    nearest_distance: float


class EvaluationResponse(BaseModel):
    model: str
    metrics: dict[str, float | int | None]
    points: list[dict[str, float | int | bool]]


class DSO3OverviewResponse(BaseModel):
    model: str
    summary: dict[str, float | int]
    pca_eigenvalues: list[float]
    pca_explained_variance_ratio: list[float]
    pca_kaiser_components: int
    points: list[dict[str, float | int | bool]]


class DSO1BatchRequest(BaseModel):
    rows: list[DSO1PredictRequest]


class DSO2BatchRequest(BaseModel):
    rows: list[DSO2PredictRequest]


class DSO3BatchRequest(BaseModel):
    rows: list[DSO3PredictRequest]
