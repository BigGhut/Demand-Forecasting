"""Stationarity testing and automatic differencing order selection.

Provides ADF, KPSS, combined verdict logic, and batch analysis across
store–family groups. Uses vectorized pandas operations where possible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import adfuller, kpss


class StationarityAnalyzer:
    """Automated stationarity testing and differencing order selection.

    Parameters
    ----------
    significance_level : float
        Threshold for p-value based stationarity decisions (default 0.05).
    """

    def __init__(self, significance_level: float = 0.05) -> None:
        if not 0.0 < significance_level < 1.0:
            raise ValueError(
                f"significance_level must be in (0, 1), got {significance_level}"
            )
        self.significance_level = significance_level

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def adf_test(self, series: pd.Series, name: str = "") -> dict[str, Any]:
        """Run the Augmented Dickey-Fuller test.

        Parameters
        ----------
        series : pd.Series
            Univariate time series (numeric, no NaNs expected).
        name : str
            Optional label used in log messages.

        Returns
        -------
        dict[str, Any]
            Keys: ``test_statistic``, ``p_value``, ``critical_values``,
            ``is_stationary``, ``lags_used``.
        """
        clean = series.dropna()
        if len(clean) < 3:
            raise ValueError(
                f"ADF requires at least 3 observations, got {len(clean)}"
            )

        stat, p_value, lags_used, _nobs, critical_values, _icbest = adfuller(
            clean, autolag="AIC"
        )

        is_stationary: bool = p_value < self.significance_level
        label = f" [{name}]" if name else ""
        logger.debug(
            f"ADF{label}: stat={stat:.4f}, p={p_value:.4f}, "
            f"lags={lags_used}, stationary={is_stationary}"
        )

        return {
            "test_statistic": float(stat),
            "p_value": float(p_value),
            "critical_values": {k: float(v) for k, v in critical_values.items()},
            "is_stationary": is_stationary,
            "lags_used": int(lags_used),
        }

    def kpss_test(self, series: pd.Series, name: str = "") -> dict[str, Any]:
        """Run the KPSS test for level-stationarity.

        Parameters
        ----------
        series : pd.Series
            Univariate time series (numeric, no NaNs expected).
        name : str
            Optional label used in log messages.

        Returns
        -------
        dict[str, Any]
            Keys: ``test_statistic``, ``p_value``, ``critical_values``,
            ``is_stationary``, ``lags_used``.

        Notes
        -----
        KPSS null hypothesis is *stationarity*, so we reject (non-stationary)
        when ``p_value < significance_level``.
        """
        clean = series.dropna()
        if len(clean) < 3:
            raise ValueError(
                f"KPSS requires at least 3 observations, got {len(clean)}"
            )

        # nlags="auto" suppresses the FutureWarning
        stat, p_value, lags_used, critical_values = kpss(
            clean, regression="c", nlags="auto"
        )

        # KPSS H0 = stationary → reject when p < alpha → non-stationary
        is_stationary: bool = p_value >= self.significance_level
        label = f" [{name}]" if name else ""
        logger.debug(
            f"KPSS{label}: stat={stat:.4f}, p={p_value:.4f}, "
            f"lags={lags_used}, stationary={is_stationary}"
        )

        return {
            "test_statistic": float(stat),
            "p_value": float(p_value),
            "critical_values": {k: float(v) for k, v in critical_values.items()},
            "is_stationary": is_stationary,
            "lags_used": int(lags_used),
        }

    # ------------------------------------------------------------------
    # Combined verdict
    # ------------------------------------------------------------------

    def combined_test(self, series: pd.Series, name: str = "") -> dict[str, Any]:
        """Run both ADF and KPSS and produce a combined verdict.

        Verdict logic
        -------------
        * Both agree **stationary** → ``"stationary"``
        * Both agree **non-stationary** → ``"non-stationary"``
        * ADF stationary, KPSS non-stationary → ``"trend-stationary"``
        * ADF non-stationary, KPSS stationary → ``"difference-stationary"``

        Returns
        -------
        dict[str, Any]
            Keys: ``adf`` (sub-dict), ``kpss`` (sub-dict), ``verdict`` (str).
        """
        adf_result = self.adf_test(series, name=name)
        kpss_result = self.kpss_test(series, name=name)

        adf_stat = adf_result["is_stationary"]
        kpss_stat = kpss_result["is_stationary"]

        if adf_stat and kpss_stat:
            verdict = "stationary"
        elif not adf_stat and not kpss_stat:
            verdict = "non-stationary"
        elif adf_stat and not kpss_stat:
            verdict = "trend-stationary"
        else:
            verdict = "difference-stationary"

        label = f" [{name}]" if name else ""
        logger.info(f"Combined test{label}: verdict={verdict}")

        return {
            "adf": adf_result,
            "kpss": kpss_result,
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # Automatic differencing order
    # ------------------------------------------------------------------

    def find_differencing_order(
        self,
        series: pd.Series,
        max_d: int = 2,
        name: str = "",
    ) -> int:
        """Find the minimum differencing order that achieves stationarity.

        Iteratively differences the series (up to *max_d* times) and checks
        the :meth:`combined_test` verdict.  A verdict of ``"stationary"`` or
        ``"trend-stationary"`` is accepted as *stationary enough*.

        Parameters
        ----------
        series : pd.Series
            Univariate time series.
        max_d : int
            Maximum differencing order to try (default 2).
        name : str
            Optional label for logging.

        Returns
        -------
        int
            Recommended differencing order ``d ∈ {0, 1, …, max_d}``.
        """
        if max_d < 0:
            raise ValueError(f"max_d must be >= 0, got {max_d}")

        current = series.copy()
        for d in range(max_d + 1):
            if d > 0:
                current = current.diff().dropna()
                if len(current) < 3:
                    logger.warning(
                        f"Series too short after {d} differencing(s); "
                        f"returning d={d - 1}"
                    )
                    return d - 1

            result = self.combined_test(current, name=f"{name} d={d}")
            if result["verdict"] in {"stationary", "trend-stationary"}:
                label = f" [{name}]" if name else ""
                logger.info(f"Differencing order{label}: d={d}")
                return d

        label = f" [{name}]" if name else ""
        logger.warning(
            f"Series{label} not stationary after d={max_d}; returning max_d"
        )
        return max_d

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_multiple(
        self,
        df: pd.DataFrame,
        group_cols: list[str],
        target_col: str = "sales",
    ) -> pd.DataFrame:
        """Run stationarity analysis on every group in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Long-form DataFrame containing *group_cols* and *target_col*.
        group_cols : list[str]
            Columns that uniquely identify each time series
            (e.g. ``["store_nbr", "family"]``).
        target_col : str
            Name of the target column to analyse.

        Returns
        -------
        pd.DataFrame
            Summary with columns:
            ``group``, ``adf_p``, ``kpss_p``, ``verdict``, ``recommended_d``.
        """
        missing = [c for c in [*group_cols, target_col] if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")

        records: list[dict[str, Any]] = []

        for group_key, group_df in df.groupby(group_cols, sort=True):
            group_label = (
                str(group_key)
                if isinstance(group_key, str)
                else " | ".join(str(k) for k in group_key)
            )
            series = group_df[target_col].reset_index(drop=True)

            if series.dropna().shape[0] < 3:
                logger.warning(
                    f"Group {group_label}: fewer than 3 non-null values, skipping"
                )
                continue

            try:
                combined = self.combined_test(series, name=group_label)
                recommended_d = self.find_differencing_order(
                    series, name=group_label
                )
            except Exception:
                logger.exception(f"Error analysing group {group_label}")
                continue

            records.append(
                {
                    "group": group_label,
                    "adf_p": combined["adf"]["p_value"],
                    "kpss_p": combined["kpss"]["p_value"],
                    "verdict": combined["verdict"],
                    "recommended_d": recommended_d,
                }
            )

        summary = pd.DataFrame.from_records(
            records,
            columns=["group", "adf_p", "kpss_p", "verdict", "recommended_d"],
        )
        logger.info(
            f"Stationarity analysis complete: {len(summary)} groups processed"
        )
        return summary
