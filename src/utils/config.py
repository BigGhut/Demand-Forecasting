"""YAML configuration loading and Pydantic validation for the forecasting pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_CONFIGS_DIR: Path = _PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Base config models
# ---------------------------------------------------------------------------


class ProphetConfig(BaseModel):
    """Prophet model hyper-parameters."""

    seasonality_mode: Literal["additive", "multiplicative"] = "multiplicative"
    changepoint_prior_scale: float = Field(0.05, gt=0)
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    holidays_country: str = "EC"


class CatBoostConfig(BaseModel):
    """CatBoost model hyper-parameters."""

    iterations: int = Field(1000, gt=0)
    learning_rate: float = Field(0.05, gt=0)
    depth: int = Field(6, ge=1, le=16)
    loss_function: str = "RMSEWithUncertainty"
    task_type: Literal["CPU", "GPU"] = "CPU"
    verbose: int = 100
    early_stopping_rounds: int = Field(50, gt=0)


class PipelineConfig(BaseModel):
    """Cross-validation and pipeline-level settings."""

    forecast_horizon: int = Field(28, gt=0)
    cv_windows: int = Field(5, gt=0)
    cv_stride: int = Field(14, gt=0)
    target_transform: Literal["difference", "log", "box_cox", "none"] = "difference"
    aggregation_level: Literal["store_family", "store_item"] = "store_family"


class BaseConfig(BaseModel):
    """Top-level base configuration container."""

    prophet: ProphetConfig = Field(default_factory=ProphetConfig)
    catboost: CatBoostConfig = Field(default_factory=CatBoostConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


# ---------------------------------------------------------------------------
# Feature config models
# ---------------------------------------------------------------------------


class RollingConfig(BaseModel):
    """Rolling-window feature parameters."""

    windows: list[int] = Field(default_factory=lambda: [7, 14, 28])
    functions: list[str] = Field(default_factory=lambda: ["mean", "std"])
    base_lags: list[int] = Field(default_factory=lambda: [28, 35])


class ExpandingConfig(BaseModel):
    """Expanding-window feature parameters."""

    functions: list[str] = Field(default_factory=lambda: ["mean", "std"])
    min_periods: int = Field(28, gt=0)


class CalendarConfig(BaseModel):
    """Calendar / date-part features."""

    features: list[str] = Field(
        default_factory=lambda: [
            "day_of_week",
            "month",
            "quarter",
            "day_of_month",
            "week_of_year",
        ],
    )
    binary: list[str] = Field(
        default_factory=lambda: [
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "is_payday",
        ],
    )


class ExternalConfig(BaseModel):
    """External signal features (oil prices, promotions, transactions)."""

    oil_lags: list[int] = Field(default_factory=lambda: [28, 35, 42])
    include_promotion: bool = True
    include_transactions: bool = True


class CategoricalConfig(BaseModel):
    """Categorical feature list."""

    features: list[str] = Field(
        default_factory=lambda: [
            "store_nbr",
            "family",
            "city",
            "state",
            "type",
            "cluster",
        ],
    )


class FeatureConfig(BaseModel):
    """Complete feature engineering configuration."""

    lags: list[int] = Field(default_factory=lambda: [28, 35, 42, 56])
    rolling: RollingConfig = Field(default_factory=RollingConfig)
    expanding: ExpandingConfig = Field(default_factory=ExpandingConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    external: ExternalConfig = Field(default_factory=ExternalConfig)
    categorical: CategoricalConfig = Field(default_factory=CategoricalConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dict.

    Parameters
    ----------
    config_path:
        Absolute or relative path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML contents.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    return data


def get_base_config(path: Path | None = None) -> BaseConfig:
    """Load and validate *base_config.yaml*.

    Parameters
    ----------
    path:
        Optional override for the config file location.
        Defaults to ``configs/base_config.yaml`` relative to the project root.
    """
    path = path or _CONFIGS_DIR / "base_config.yaml"
    raw = load_config(path)
    return BaseConfig.model_validate(raw)


def get_feature_config(path: Path | None = None) -> FeatureConfig:
    """Load and validate *feature_config.yaml*.

    Parameters
    ----------
    path:
        Optional override for the config file location.
        Defaults to ``configs/feature_config.yaml`` relative to the project root.
    """
    path = path or _CONFIGS_DIR / "feature_config.yaml"
    raw = load_config(path)
    return FeatureConfig.model_validate(raw)
