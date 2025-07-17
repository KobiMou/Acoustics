# NoLeak Data Analysis Issues and Solutions

## Issues Identified

### 1. Non-Zero SNR_Distance_dB for NoLeak Data

**Problem**: NoLeak measurements are showing non-zero SNR_Distance_dB values instead of the expected 0 dB.

**Root Cause**: The current distance-specific SNR calculation (lines 974-981 in main_11.py) calculates SNR for **all measurements** at a given distance, including NoLeak measurements themselves, against the noise floor created from NoLeak measurements at that same distance.

**Current Logic**:
```python
# Create distance-specific noise floor from NoLeak measurements
distance_noise_floor = np.mean(np.vstack([data['psd_avg'] for data in group['noleak']]), axis=0)

# Calculate SNR for ALL measurements (including NoLeak)
for data in group['all_measurements']:  # This includes NoLeak measurements!
    snr_distance_dB = 10 * np.log10(data['psd_avg'] / distance_noise_floor)
```

**Why This Causes Issues**:
- When there are multiple NoLeak measurements at the same distance, each individual NoLeak measurement is compared against the average of all NoLeak measurements
- This creates non-zero SNR values for NoLeak data because individual measurements naturally deviate from their group average
- The deviation becomes more pronounced with measurement noise and slight variations between recordings

### 2. Baseline Count Shows 2 Instead of 1

**Problem**: The baseline_measurements count is showing 2 instead of the expected 1 for some distance groups.

**Root Cause**: The baseline count (line 656) correctly counts the number of NoLeak measurements available for that distance:
```python
detection_result['baseline_measurements'] = len(group['noleak'])
```

**Analysis**: If the count is 2, it means there are actually 2 NoLeak measurements at that distance. This could be due to:
- Multiple NoLeak segments extracted from different files at the same distance
- Files with similar naming patterns being classified as NoLeak
- Duplicate processing of the same data

## Solutions

### Solution 1: Fix SNR_Distance_dB for NoLeak Measurements

**Approach**: NoLeak measurements should have SNR_Distance_dB = 0 dB by definition, since they represent the baseline noise floor.

**Implementation**:
```python
# Calculate SNR for all measurements at this distance
for data in group['all_measurements']:
    if data['is_noleak']:
        # NoLeak measurements should have 0 dB SNR against their own baseline
        snr_distance_dB = np.zeros_like(data['psd_avg'])
    else:
        # Calculate distance-specific SNR in dB for potential leak measurements
        snr_distance_dB = 10 * np.log10(data['psd_avg'] / distance_noise_floor)
```

### Solution 2: Investigate Baseline Count Issue

**Diagnostic Steps**:
1. Add logging to show which files are being classified as NoLeak at each distance
2. Verify the filename patterns and ensure correct classification
3. Check for duplicate processing or multiple segments from the same source

**Implementation**:
```python
# Add debugging information
print(f"Distance {distance}m - NoLeak files: {[data['filename'] for data in group['noleak']]}")
print(f"Distance {distance}m - Baseline count: {len(group['noleak'])}")
```

### Solution 3: Enhanced NoLeak Classification

**Approach**: Improve the NoLeak detection logic to be more robust and prevent false positives.

**Current Logic**:
```python
'is_noleak': 'noleak' in ListFileNames[iLoop].lower()
```

**Enhanced Logic**:
```python
def is_noleak_measurement(filename):
    """
    Determine if a measurement is a NoLeak baseline measurement
    """
    filename_lower = filename.lower()
    
    # Check for explicit noleak indicators
    noleak_indicators = ['_noleak', 'noleak_', 'no_leak', 'baseline', 'background']
    
    for indicator in noleak_indicators:
        if indicator in filename_lower:
            return True
    
    # Check for leak indicators that would override noleak
    leak_indicators = ['_leak', 'leak_', 'leakage']
    for indicator in leak_indicators:
        if indicator in filename_lower:
            return False
    
    return False
```

## Recommended Implementation Order

1. **Immediate Fix**: Implement Solution 1 to set SNR_Distance_dB = 0 for NoLeak measurements
2. **Diagnostic**: Add logging from Solution 2 to understand the baseline count issue
3. **Long-term**: Consider implementing Solution 3 for more robust classification

## Expected Results After Fix

- NoLeak measurements will show SNR_Distance_dB = 0 dB consistently
- Baseline count will reflect the actual number of NoLeak measurements
- Distance-specific analysis will be more accurate and interpretable
- Charts will show NoLeak reference lines at 0 dB as expected

## Files to Modify

- `main_11.py`: Lines 974-981 (SNR calculation logic)
- `main_11.py`: Lines 380-390 (NoLeak classification logic)
- Add diagnostic logging around lines 960-970 (distance grouping)