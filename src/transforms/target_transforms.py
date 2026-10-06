"""Reversible target variable transformations for time series forecasting.

Each transformation stores enough state to guarantee exact inversion.
A round-trip test is executed automatically during :meth:`fit_transform`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import inv_boxcox
from scipy.stats import boxcox

TransformMethod = Literal["difference", "log", "box_cox", "none"]

_VALID_METHODS: frozenset[str] = frozenset({"difference", "log", "box_cox", "none"})


class TargetTransformer:
    """Reversible target transformations for time series.

    Supported methods
    -----------------
    * ``"difference"`` — first-order differencing (``diff(1)`` / ``cumsum``).
    * ``"log"`` — ``log1p`` / ``expm1`` (safe for zeros).
    * ``"box_cox"`` — Box-Cox with λ fit on the series passed to :meth:`fit`.
      If that series has a non-positive value, add 1 to every observation
      (not only to the zeros). Each training window needs its own :meth:`fit`.
    * ``"none"`` — identity pass-through.

    **Critical invariant**: ``inverse_transform(transform(x)) ≈ x``.
    This is verified automatically inside :meth:`fit_transform`.

    Parameters
    ----------
    method : TransformMethod
        Transformation method to apply.
    """

    def __init__(self, method: TransformMethod = "difference") -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {sorted(_VALID_METHODS)}."
            )
        self.method: TransformMethod = method
        self._is_fitted: bool = False

        # State stored during fit (depending on method)
        self._first_value: float | None = None
        self._boxcox_lambda: float | None = None
        self._boxcox_shift: float = 0.0
        self._boxcox_fit_size: int | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, series: pd.Series) -> TargetTransformer:
        """Fit the transformer, storing state needed for inversion.

        Parameters
        ----------
        series : pd.Series
            Training target for the current window. Box-Cox λ is estimated
            from this sample only and is replaced on the next ``fit`` call.

        Returns
        -------
        TargetTransformer
            ``self``, for method chaining.
        """
        if series.dropna().empty:
            raise ValueError("Cannot fit on an empty or all-NaN series.")

        if self.method == "difference":
            self._first_value = float(series.iloc[0])
            logger.debug(f"Differencing fit: first_value={self._first_value}")

        elif self.method == "log":
            if (series.dropna() < 0).any():
                raise ValueError(
                    "Log transform (log1p) requires non-negative values."
                )
            logger.debug("Log transform fitted (stateless aside from validation).")

        elif self.method == "box_cox":
            # λ is the Gaussian MLE on this series alone. Call fit again for
            # the next CV window; do not reuse a λ estimated on a longer sample.
            clean = series.dropna().astype(float)
            if (clean <= 0).any():
                # One constant for every timestamp. Adding 1 only on zero days
                # would turn both an original 0 and an original 1 into 1.
                self._boxcox_shift = 1.0
                logger.warning(
                    "Box-Cox input has non-positive values. Adding 1 to every observation."
                )
            else:
                self._boxcox_shift = 0.0
            shifted = clean.to_numpy(dtype=float) + self._boxcox_shift
            if np.any(shifted <= 0):
                raise ValueError(
                    "Box-Cox needs strictly positive values after adding 1. "
                    "Clip the target at 0 before fitting."
                )
            _, lam = boxcox(shifted)
            self._boxcox_lambda = float(lam)
            self._boxcox_fit_size = int(clean.shape[0])
            logger.debug(
                f"Box-Cox fit on {self._boxcox_fit_size} points: "
                f"lambda={self._boxcox_lambda:.6f}, shift={self._boxcox_shift}"
            )

        else:
            logger.debug("Identity transform (none) — no fitting needed.")

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Forward transform
    # ------------------------------------------------------------------

    def transform(self, series: pd.Series) -> pd.Series:
        """Apply the forward transformation.

        Parameters
        ----------
        series : pd.Series
            Target series to transform.

        Returns
        -------
        pd.Series
            Transformed series (may be shorter for differencing).
        """
        self._check_fitted()

        if self.method == "difference":
            return series.diff().iloc[1:]  # drop first NaN

        if self.method == "log":
            return pd.Series(
                np.log1p(series.values), index=series.index, name=series.name
            )

        if self.method == "box_cox":
            assert self._boxcox_lambda is not None
            shifted = series.astype(float) + self._boxcox_shift
            transformed = boxcox(shifted.values, lmbda=self._boxcox_lambda)
            return pd.Series(transformed, index=series.index, name=series.name)

        # method == "none"
        return series.copy()

    # ------------------------------------------------------------------
    # Inverse transform
    # ------------------------------------------------------------------

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        """Undo the transformation.

        Satisfies ``inverse_transform(transform(x)) ≈ x``.

        Parameters
        ----------
        series : pd.Series
            Previously transformed series.

        Returns
        -------
        pd.Series
            Reconstructed original-scale series.
        """
        self._check_fitted()

        if self.method == "difference":
            assert self._first_value is not None
            # Prepend the stored first value, then cumsum to invert diff
            full = pd.concat(
                [
                    pd.Series(
                        [self._first_value],
                        index=[series.index[0] - 1]
                        if isinstance(series.index, pd.RangeIndex)
                        else [series.index[0]],
                    ),
                    series,
                ]
            )
            reconstructed = full.cumsum()
            # Return with the original transformed-series index length
            return reconstructed.iloc[1:].reset_index(drop=True)

        if self.method == "log":
            return pd.Series(
                np.expm1(series.values), index=series.index, name=series.name
            )

        if self.method == "box_cox":
            assert self._boxcox_lambda is not None
            inv = inv_boxcox(series.values, self._boxcox_lambda) - self._boxcox_shift
            return pd.Series(inv, index=series.index, name=series.name)

        # method == "none"
        return series.copy()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def fit_transform(self, series: pd.Series) -> pd.Series:
        """Fit on *series*, transform it, then run the round-trip test.

        Parameters
        ----------
        series : pd.Series
            Original target series.

        Returns
        -------
        pd.Series
            Transformed series.

        Raises
        ------
        AssertionError
            If the round-trip test fails.
        """
        self.fit(series)
        transformed = self.transform(series)

        # Automatic round-trip verification
        passed = self.round_trip_test(series)
        if passed:
            logger.info(
                f"Round-trip test PASSED for method='{self.method}'"
            )
        # If it didn't pass, round_trip_test already raised AssertionError.

        return transformed

    # ------------------------------------------------------------------
    # Round-trip verification
    # ------------------------------------------------------------------

    def round_trip_test(
        self, series: pd.Series, atol: float = 1e-6
    ) -> bool:
        """Verify ``inverse_transform(transform(series)) ≈ series``.

        Parameters
        ----------
        series : pd.Series
            Original (untransformed) series.
        atol : float
            Absolute tolerance for element-wise comparison.

        Returns
        -------
        bool
            ``True`` if the round-trip is within tolerance.

        Raises
        ------
        AssertionError
            When the maximum absolute error exceeds *atol*, with
            diagnostic information logged.
        """
        self._check_fitted()

        transformed = self.transform(series)
        reconstructed = self.inverse_transform(transformed)

        # For differencing, the transformed series is 1 element shorter,
        # so we compare against the original *without* the first element.
        if self.method == "difference":
            original_aligned = series.iloc[1:].reset_index(drop=True)
        else:
            original_aligned = series.reset_index(drop=True)

        reconstructed_aligned = reconstructed.reset_index(drop=True)

        abs_error = np.abs(
            original_aligned.values.astype(np.float64)
            - reconstructed_aligned.values.astype(np.float64)
        )
        max_error = float(np.nanmax(abs_error)) if len(abs_error) > 0 else 0.0
        mean_error = float(np.nanmean(abs_error)) if len(abs_error) > 0 else 0.0

        if max_error > atol:
            msg = (
                f"Round-trip test FAILED for method='{self.method}': "
                f"max_error={max_error:.2e}, mean_error={mean_error:.2e}, "
                f"atol={atol:.2e}"
            )
            logger.error(msg)
            raise AssertionError(msg)

        logger.debug(
            f"Round-trip OK (method='{self.method}'): "
            f"max_error={max_error:.2e}, mean_error={mean_error:.2e}"
        )
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if :meth:`fit` has not been called."""
        if not self._is_fitted:
            raise RuntimeError(
                "Transformer has not been fitted. Call fit() or fit_transform() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        extras = ""
        if self._is_fitted:
            if self.method == "difference":
                extras = f", first_value={self._first_value}"
            elif self.method == "box_cox":
                extras = f", lambda={self._boxcox_lambda:.6f}, shift={self._boxcox_shift}"
        return f"TargetTransformer(method='{self.method}', {status}{extras})"
