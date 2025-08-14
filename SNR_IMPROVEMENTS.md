# SNR Calculation Improvements

## Overview

This document explains the improvements made to the SNR (Signal-to-Noise Ratio) calculation methods in the audio analysis system to address the issue of artificially high SNR values caused by near-zero noise floor values.

## The Problem

The original SNR calculation used a simple division:
```python
snr = signal / noise_floor
```

This approach had several issues:
1. **Near-zero noise values**: When noise floor values were very close to zero, division resulted in extremely high (unrealistic) SNR values
2. **No validation**: No checks for invalid or unrealistic SNR values
3. **Simple noise floor estimation**: Used only mean values, which are sensitive to outliers
4. **No safeguards**: No clipping or bounds on SNR values

## The Solution

### 1. Robust Noise Floor Estimation

The new `improved_noise_floor_estimation()` function provides multiple methods for calculating noise floor:

#### Available Methods:
- **`percentile`** (default): Uses the 10th percentile, more robust against outliers
- **`median`**: Uses median, good for smaller sample sizes
- **`mean`**: Traditional mean (original method)
- **`trimmed_mean`**: Removes top/bottom 10% then takes mean
- **`robust_avg`**: Median + small offset for extra robustness

#### Key Features:
- **Minimum threshold**: Enforces a minimum noise floor value (default: 1e-12) to prevent division by very small numbers
- **Automatic method selection**: Different methods for global vs. file-specific calculations

### 2. Robust SNR Calculation

The new `calculate_robust_snr()` function provides multiple calculation methods:

#### Available Methods:
- **`clipped_linear`** (default): Linear SNR clipped to reasonable range
- **`clipped_db`**: dB SNR clipped to reasonable range
- **`linear`**: Traditional linear SNR (original method)
- **`db`**: dB SNR with safe log conversion
- **`log10`**: Log10 scale SNR

#### Key Features:
- **Automatic clipping**: Prevents unrealistic values (default: -40 to +60 dB range)
- **Safe logarithm**: Handles zero/negative values gracefully
- **Multiple scales**: Choose between linear, dB, or log10 scales

### 3. SNR Validation and Outlier Detection

The new `validate_snr_calculation()` function provides comprehensive validation:

#### Validation Checks:
- **Very low noise floor detection**: Warns when noise floor < 1e-10
- **Unrealistic SNR values**: Detects SNR > 1000 (linear scale)
- **Non-finite values**: Checks for NaN/Inf values
- **Statistical summary**: Provides mean, median, std, min, max statistics

#### Outlier Detection Methods:
- **`iqr`** (default): Interquartile range method
- **`zscore`**: Z-score based detection
- **`percentile`**: Percentile-based bounds

## Configuration

All SNR parameters are now configurable through the `SNR_CONFIG` dictionary at the top of `main_11.py`:

```python
SNR_CONFIG = {
    'noise_floor_method': 'percentile',     # Noise floor estimation method
    'noise_floor_percentile': 10,          # Percentile for percentile method
    'min_noise_threshold': 1e-12,          # Minimum noise floor value
    'snr_calculation_method': 'clipped_linear',  # SNR calculation method
    'max_snr_db': 60,                      # Maximum SNR in dB
    'min_snr_db': -40,                     # Minimum SNR in dB
    'file_specific_method': 'median',      # Method for file-specific calculations
    'enable_snr_validation': True,         # Enable validation warnings
    'outlier_detection_method': 'iqr',     # Outlier detection method
    'outlier_threshold_factor': 3.0        # Outlier detection threshold
}
```

## Usage Examples

### Example 1: Default Robust Calculation
```python
# Automatically uses percentile method with 10th percentile and clipping
noise_floor = improved_noise_floor_estimation(noleak_data)
snr = calculate_robust_snr(signal, noise_floor)
```

### Example 2: Conservative Approach
```python
# For very noisy data, use higher minimum threshold and median
SNR_CONFIG['min_noise_threshold'] = 1e-10
SNR_CONFIG['noise_floor_method'] = 'median'
SNR_CONFIG['max_snr_db'] = 40  # Lower maximum
```

### Example 3: Traditional Approach (for comparison)
```python
# Revert to original behavior
SNR_CONFIG['noise_floor_method'] = 'mean'
SNR_CONFIG['snr_calculation_method'] = 'linear'
SNR_CONFIG['min_noise_threshold'] = 1e-16  # Very small threshold
```

### Example 4: dB Scale Output
```python
# Get SNR in dB scale with clipping
SNR_CONFIG['snr_calculation_method'] = 'clipped_db'
# Results will be in dB, clipped to [-40, 60] dB range
```

## Recommended Settings by Use Case

### 1. General Audio Analysis (Default)
```python
'noise_floor_method': 'percentile'
'snr_calculation_method': 'clipped_linear'
'max_snr_db': 60
'min_snr_db': -40
```

### 2. Very Low Noise Environment
```python
'noise_floor_method': 'robust_avg'
'min_noise_threshold': 1e-14
'snr_calculation_method': 'clipped_db'
'max_snr_db': 80
```

### 3. High Noise Environment
```python
'noise_floor_method': 'median'
'min_noise_threshold': 1e-10
'max_snr_db': 40
'min_snr_db': -20
```

### 4. Research/Publication Quality
```python
'enable_snr_validation': True
'outlier_detection_method': 'iqr'
'snr_calculation_method': 'clipped_db'  # dB scale for publications
```

## Benefits of the Improved System

1. **Eliminates Unrealistic SNR Values**: No more infinite or extremely high SNR values due to near-zero noise
2. **Configurable**: Easy to adjust parameters for different use cases
3. **Robust**: Multiple estimation methods handle various data characteristics
4. **Validated**: Automatic detection of potential issues with recommendations
5. **Backward Compatible**: Can revert to original behavior if needed
6. **Well Documented**: Clear understanding of what each parameter does

## Troubleshooting

### Issue: Still getting very high SNR values
**Solution**: 
- Increase `min_noise_threshold` (e.g., to 1e-10)
- Use `clipped_linear` or `clipped_db` method
- Lower `max_snr_db` value

### Issue: SNR values seem too low
**Solution**:
- Check if `min_noise_threshold` is too high
- Try `mean` method instead of `percentile`
- Verify your noise measurements are valid

### Issue: Too many validation warnings
**Solution**:
- Set `enable_snr_validation: False` to disable warnings
- Adjust `outlier_threshold_factor` to be less sensitive
- Review your input data quality

### Issue: Need traditional behavior
**Solution**:
```python
SNR_CONFIG['noise_floor_method'] = 'mean'
SNR_CONFIG['snr_calculation_method'] = 'linear'
SNR_CONFIG['enable_snr_validation'] = False
```

## Technical Details

### Noise Floor Estimation Formulas

**Percentile Method (Default):**
```python
noise_floor = np.percentile(noise_stack, percentile, axis=0)
noise_floor = np.maximum(noise_floor, min_noise_threshold)
```

**Robust Average Method:**
```python
median_noise = np.median(noise_stack, axis=0)
min_noise = np.min(noise_stack, axis=0)
noise_floor = median_noise + 0.1 * (median_noise - min_noise)
```

### SNR Calculation Formulas

**Clipped Linear (Default):**
```python
snr_linear = signal / noise_floor
max_snr_linear = 10**(max_snr_db/10)
min_snr_linear = 10**(min_snr_db/10)
snr = np.clip(snr_linear, min_snr_linear, max_snr_linear)
```

**Clipped dB:**
```python
snr_db = 10 * log10(signal / noise_floor)
snr = np.clip(snr_db, min_snr_db, max_snr_db)
```

## Migration Guide

If you have existing analysis results and want to compare with the new method:

1. **Save current configuration** before making changes
2. **Run both methods** on the same data for comparison
3. **Gradually adjust parameters** to match your specific needs
4. **Validate results** using the built-in validation functions

The improved SNR calculation system provides much more reliable and realistic SNR values while maintaining flexibility for different use cases.