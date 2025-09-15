from typing import Optional, Dict, Tuple, Union, Any
from scipy.signal import savgol_filter
import numpy as np
import json
import numba
import logging
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@numba.njit(parallel=True)
def _mask_interpolate_2D(band_data_2D: np.ndarray) -> np.ndarray:
    """
    Interpolates missing values (np.nan) in a 2D array along axis=1 (time).
    
    Args:
        band_data_2D (np.ndarray): 2D array with shape (samples, time).
        
    Returns:
        np.ndarray: Interpolated 2D array of the same shape as input.
    """
    n_samples, n_timestamps = band_data_2D.shape
    band_interp_data_2D = band_data_2D.copy()

    for n_sample in numba.prange(n_samples):
        valid_indexes = np.where(~np.isnan(band_data_2D[n_sample]))[0]
        if valid_indexes.size == 0:
            band_interp_data_2D[n_sample] = 0  
            continue
        if valid_indexes.size == n_timestamps:
            continue
        fp = band_data_2D[n_sample, valid_indexes]
        band_interp_data_2D[n_sample] = np.interp(
            np.arange(n_timestamps),
            valid_indexes,
            fp
        )
    return band_interp_data_2D


def _flatten_2D(arr: np.ndarray) -> np.ndarray:
    """
    Flattens an N-dimensional array into 2D by collapsing all but last dimension.

    Args:
        arr (np.ndarray): Input array of shape (n1, n2, ..., nk, n_t).

    Returns:
        np.ndarray: 2D array of shape (n1*n2*...*nk, n_t).
    """
    *n_rem, n_ts = arr.shape
    return arr.reshape((np.prod(n_rem), n_ts))


def mask_interpolate(
    band_data: np.ndarray,
    mask_value: float = np.nan,
    n_jobs: int = 1
) -> np.ndarray:
    """
    Interpolates missing values (np.nan and mask_value) across time in an N-D array.

    Args:
        band_data (np.ndarray): Array with shape (samples, time) or similar.
        mask_value (float): Value considered as missing (default 65535).
        n_jobs (int): Number of threads for parallelization; -1 to use all CPUs.

    Returns:
        np.ndarray: Same shape as input with missing values interpolated.
    """
    band_data = np.array(band_data, dtype=np.float32)
    logger.info(f"Count of NaNs before replacement: {np.isnan(band_data).sum()}")
    logger.info(f"mask_value: {mask_value}")
    if np.isnan(mask_value):
        pass
    else:
        band_data = band_data.astype(np.float32)
        band_data = np.where(band_data == mask_value, np.nan, band_data)
    if not np.isnan(band_data).any():
        return band_data
    band_data_2D = _flatten_2D(band_data)
    
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    numba.set_num_threads(n_jobs)
    logger.info(f"Count of NaNs before interp: {np.isnan(band_data_2D).sum()}")
    band_data_2D_interp = _mask_interpolate_2D(band_data_2D)
    logger.info(f"Count of NaNs after interp: {np.isnan(band_data_2D_interp).sum()}")
    return band_data_2D_interp.reshape(band_data.shape)

class Preprocess:
    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[str] = None,
        outlier_dict: Optional[Dict] = None,
        normalise_dict: Optional[Dict] = None,
        mode: Optional[str] = None,
        MODEL_DIR: Optional[str] = None,
    ):
        self.data = data
        self.labels = labels
        self.outlier_dict = outlier_dict if outlier_dict is not None else {}
        self.normalise_dict = normalise_dict if normalise_dict is not None else {}
        self.mode = mode
        self.MODEL_DIR = MODEL_DIR

    def remove_outliers(self, band_index: int, band_data, per_timestamp: bool = False, mask_value: int = 65535) -> None:
        """
        Removes outliers by setting values outside IQR bounds to NaN.
        Can operate globally across all timestamps or per timestamp.

        Args:
            band_index (int): Index of the band to process.
            per_timestamp (bool): If True, detect outliers per timestamp.
                                  If False, detect globally across all timestamps.
        """
        if self.mode == 'train':
            if not per_timestamp:
                # Global outlier removal
                q1 = np.nanpercentile(band_data, 25)
                q3 = np.nanpercentile(band_data, 75)
                iqr = q3 - q1
                lb = q1 - 1.5 * iqr
                ub = q3 + 1.5 * iqr

                mask_outliers = (band_data < lb) | (band_data > ub)
                band_data[mask_outliers] = mask_value

                self.outlier_dict[band_index] = [float(round(lb, 2)), float(round(ub, 2))]

            else:
                print("Per timestamp outlier removal")
                n_timestamps = band_data.shape[1]
                band_outliers = {}
                for t in range(n_timestamps):
                    vals = band_data[:, t]
                    q1 = np.nanpercentile(vals, 25)
                    q3 = np.nanpercentile(vals, 75)
                    iqr = q3 - q1
                    lb = q1 - 1.5 * iqr
                    ub = q3 + 1.5 * iqr

                    mask_outliers = (vals < lb) | (vals > ub)
                    n_masked = np.sum(mask_outliers)
                    print(f"Timestep {t}: masked {n_masked} outliers")

                    band_data[mask_outliers, t] = np.nan
                    band_outliers[t] = [float(round(lb, 2)), float(round(ub, 2))]
                self.outlier_dict[band_index] = band_outliers

        elif self.mode == 'inference':
            sample_key = next(iter(self.outlier_dict))
            key_type = type(sample_key)
            band_index_key = key_type(band_index)
            if not per_timestamp:
                lb, ub = self.outlier_dict[band_index_key]
                mask_outliers = (band_data < lb) | (band_data > ub)
                band_data[mask_outliers] = np.nan
                # self.data[:, :, band_index] = band_data
            else:
                band_outliers = self.outlier_dict[band_index_key]
                for t in range(band_data.shape[1]):
                    lb, ub = band_outliers.get(t, (None, None))
                    if lb is None or ub is None:
                        continue
                    vals = band_data[:, t]
                    mask_outliers = (vals < lb) | (vals > ub)
                    band_data[mask_outliers, t] = np.nan
                # self.data[:, :, band_index] = band_data

        return band_data
    
    def resample_time_series(self, data, original_interval_days=20, target_interval_days=10):
        """
        Resamples 3D time series data from a coarser interval to a finer one using interpolation.

        Parameters:
            data (np.ndarray): Shape (n_samples, n_timepoints, n_features)
            original_interval_days (int): Original time step in days
            target_interval_days (int): Desired time step in days

        Returns:
            np.ndarray: Resampled data with shape (n_samples, new_timepoints, n_features)
        """
        n_samples, n_original_steps, n_features = data.shape
        n_target_steps = int(n_original_steps * (original_interval_days / target_interval_days))

        original_times = np.arange(n_original_steps) * original_interval_days
        target_times = np.linspace(original_times[0], original_times[-1], n_target_steps)

        resampled_data = np.empty((n_samples, n_target_steps, n_features))

        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                interpolator = interp1d(original_times, data[sample_idx, :, feature_idx], kind='linear')
                resampled_data[sample_idx, :, feature_idx] = interpolator(target_times)

        return resampled_data
    
    def normalise_features(self, band_data, band_index: int) -> None:
        """
        Normalizes the feature band data to the range [-1, 1].

        Args:
            band_index (int): Index of the band to normalize.
        """

        if self.mode == 'train':
            finite_vals = band_data[np.isfinite(band_data)]
            lb = np.min(finite_vals)
            ub = np.max(finite_vals)
            denom = ub - lb if ub != lb else 1e-6

            normed = ((2 * (band_data - lb)) / denom) - 1
            self.normalise_dict[band_index] = [float(round(lb, 2)), float(round(ub, 2))]

        elif self.mode == 'inference':
            sample_key = next(iter(self.normalise_dict))
            key_type = type(sample_key)
            band_index_key = key_type(band_index)
            lb, ub = self.normalise_dict[band_index_key]
            denom = ub - lb if ub != lb else 1e-6
            normed = ((2 * (band_data - lb)) / denom) - 1
        return normed

    def prep(
        self,
        outlier_per_timestamp: bool = False,
        mask_value: float = np.nan
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Runs outlier removal, interpolation, smoothing, and normalization on all bands.

        Args:
            outlier_per_timestamp (bool): If True, do outlier detection per timestamp.
            mask_value (float): Value considered as missing.

        Returns:
            Tuple[np.ndarray, Dict, Dict]: Processed data, outlier dict, normalization dict.
        """
        import time

        n_samples, n_timestamps, n_bands = self.data.shape
        processed_data = np.empty((n_samples, n_timestamps, n_bands), dtype=np.float32)

        for band_index in range(n_bands):
            logger.info(f"Preprocessing. Band : {band_index}")

            # # Outlier removal
            t0 = time.time()
            band_data = self.data[:, :, band_index].copy()
            band_data = self.remove_outliers(
                band_index=band_index,
                band_data=band_data,
                per_timestamp=outlier_per_timestamp,
                mask_value=mask_value,
            )

            logger.info(f"Outliers clipped. Method : {outlier_per_timestamp}")
            logger.info(f"Outliers took {time.time() - t0:.2f} seconds")

            # Interpolation of missing values
            t0 = time.time()
            logger.info(f"Masking value: {mask_value}")
            logger.info(f"Pre-interp min: {np.nanmin(band_data)}, max: {np.nanmax(band_data)}")
            band_data = mask_interpolate(band_data, mask_value)
            logger.info(f"Post-interp min: {np.nanmin(band_data)}, max: {np.nanmax(band_data)}")
            logger.info(f"Missing values interpolated")
            logger.info(f"Interpolation took {time.time() - t0:.2f} seconds")

            # Smoothing using Savitzky-Golay
            t0 = time.time()
            band_data = savgol_filter(
                band_data.astype(np.float32), 7, 2, axis=1, mode='nearest'
            )
            logger.info(f"Post-smooth min: {band_data.min()}, max: {band_data.max()}")
            logger.info(f"Smoothed with Sav-Gol (window=7, poly=2)")
            logger.info(f"Smoothing took {time.time() - t0:.2f} seconds")

            # Normalisation
            t0 = time.time()
            band_data = self.normalise_features(band_data, band_index)
            logger.info(f"Post-norm min: {band_data.min()}, max: {band_data.max()}")
            logger.info(f"Normalized to [-1, 1]")
            logger.info(f"Normalisation took {time.time() - t0:.2f} seconds")

            # Save final result for this band
            processed_data[:, :, band_index] = band_data
            print("\n\n")

        # Save metadata if in training mode
        if self.mode == 'train' and self.MODEL_DIR is not None:
            with open(f"{self.MODEL_DIR}/outlier_dict.json", 'w') as f_out:
                json.dump(self.outlier_dict, f_out)
            with open(f"{self.MODEL_DIR}/normalise_dict.json", 'w') as f_out:
                json.dump(self.normalise_dict, f_out)

        # Clean up memory
        del self.data
        self.data = None
        return processed_data, self.outlier_dict, self.normalise_dict