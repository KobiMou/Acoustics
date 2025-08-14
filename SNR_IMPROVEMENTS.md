# SNR Calculation Improvements

## Problem Description

The original codebase had several issues with Signal-to-Noise Ratio (SNR) calculations that could lead to unrealistic or misleading results:

1. **Extreme SNR values**: When noise floor values approached zero, SNR calculations could produce extremely high values (approaching infinity)
2. **Machine epsilon protection**: Using `np.finfo(float).eps` (≈2.22e-16) as minimum denominator was too small for practical acoustics applications
3. **No upper bounds**: SNR values could reach impractical levels (e.g., 1000+ dB) that don't reflect real-world measurement limitations
4. **Inconsistent handling**: Different parts of the code used different approaches to handle near-zero denominators

## Implemented Solutions

### 1. Robust SNR Calculation Function

Added `calculate_robust_snr()` function with four different methods:

#### **Adaptive Method** (Default)
- Uses statistical properties of the noise floor
- Sets minimum noise as `max(mean - 2×std, 1% of mean)`
- Automatically adapts to data characteristics

#### **Percentile Method**
- Uses percentile-based minimum noise threshold
- Default: 10th percentile of noise floor
- Good for datasets with outliers

#### **Hybrid Method**
- Combines percentile and adaptive approaches
- Uses `max(10th_percentile, 1% of mean)`
- Most conservative approach

#### **Capped Method**
- Simple approach using fixed fraction (10%) of mean noise
- Fastest computation
- Good for real-time applications

### 2. Robust SNR Ratio Function

Added `calculate_robust_snr_ratio()` for frequency band analysis:

#### **Statistical Method** (Default)
- Uses baseline standard deviation for intelligent thresholding
- Sets minimum baseline as `max(mean - 2×std, 10% of mean)`
- Provides confidence metrics

#### **Conservative Method**
- Uses 10% of baseline mean as minimum
- Simple and reliable

#### **Adaptive Method**
- Adapts minimum baseline based on signal level
- Prevents extreme ratios by using 1% of signal as minimum

### 3. Key Features

#### **SNR Capping**
- Default maximum: 60 dB (1,000,000:1 ratio)
- Configurable upper limits
- Tracks where capping occurred

#### **Confidence Metrics**
- Coefficient of variation analysis
- Confidence levels: high, medium, low
- Helps interpret reliability of results

#### **Metadata Tracking**
- Records effective vs original noise floors
- Tracks capping statistics
- Documents method used

## Updated Code Locations

### Global SNR Calculations (Lines ~740-744)
```python
# Old approach
snr = data['psd_avg'] / noise_floor
snr_fft = data['fft_abs_min'] / fft_noise_floor

# New robust approach
snr_result = calculate_robust_snr(data['psd_avg'], noise_floor, method='adaptive', max_snr_db=60)
snr = snr_result['snr_linear']
snr_fft_result = calculate_robust_snr(data['fft_abs_min'], fft_noise_floor, method='adaptive', max_snr_db=60)
snr_fft = snr_fft_result['snr_linear']
```

### File-Specific SNR Calculations (Lines ~1150-1154)
```python
# Old approach
snr_file = data['psd_avg'] / wav_file_noise_floor
snr_fft_file = data['fft_abs_min'] / wav_file_fft_noise_floor

# New robust approach
snr_file_result = calculate_robust_snr(data['psd_avg'], wav_file_noise_floor, method='adaptive', max_snr_db=60)
snr_file = snr_file_result['snr_linear']
snr_fft_file_result = calculate_robust_snr(data['fft_abs_min'], wav_file_fft_noise_floor, method='adaptive', max_snr_db=60)
snr_fft_file = snr_fft_file_result['snr_linear']
```

### Frequency Band Analysis (Lines ~632-634)
```python
# Old approach
snr_ratio = leak_mean / np.maximum(baseline_mean, np.finfo(float).eps)

# New robust approach
snr_ratio_result = calculate_robust_snr_ratio(leak_mean, baseline_mean, baseline_std, method='statistical')
snr_ratio = snr_ratio_result['snr_ratio']
```

### Power Ratio Detection (Lines ~638-642)
```python
# Old approach
baseline_power = np.maximum(baseline_power, np.finfo(float).eps)
power_ratio_dB = 10 * np.log10(leak_psd / baseline_power)

# New robust approach
snr_result = calculate_robust_snr(leak_psd, baseline_power, method='adaptive', max_snr_db=40)
power_ratio_dB = snr_result['snr_db']
```

## Benefits

1. **Realistic SNR Values**: Caps prevent unrealistic infinite or extremely high SNR values
2. **Context-Aware Thresholds**: Minimum noise floors adapt to actual data characteristics
3. **Improved Reliability**: Confidence metrics help interpret result quality
4. **Configurable Limits**: Maximum SNR limits can be adjusted based on application needs
5. **Better Documentation**: Detailed metadata for troubleshooting and validation
6. **Consistent Handling**: Uniform approach across all SNR calculations

## Usage Recommendations

### For Acoustic Leak Detection:
- Use `adaptive` method for general analysis
- Set `max_snr_db=60` for typical environmental acoustics
- Use `statistical` method for frequency band analysis

### For High-Precision Measurements:
- Use `hybrid` method for conservative results
- Set lower `max_snr_db` (e.g., 40) for more realistic limits
- Monitor confidence metrics

### For Real-Time Applications:
- Use `capped` method for fastest computation
- Adjust `min_noise_percentile` based on noise characteristics

## Configuration Parameters

- `max_snr_db`: Maximum SNR in dB (default: 60)
- `min_noise_percentile`: Percentile for minimum threshold (default: 10)
- `method`: Calculation method ('adaptive', 'percentile', 'hybrid', 'capped')
- `max_ratio`: Maximum linear ratio for frequency bands (default: 1000)

## Backward Compatibility

The improvements maintain the same output structure and column names in Excel files, ensuring compatibility with existing analysis workflows while providing more robust and meaningful SNR values.