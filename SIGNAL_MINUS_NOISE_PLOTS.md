# Signal Minus Noise Plots Implementation

## Overview

Added comprehensive signal minus noise plotting functionality to complement the existing SNR analysis. These plots show the absolute magnitude of the signal above the noise floor, providing an alternative view of leak detection performance.

## New Plot Types

### 1. Global Signal Minus Noise vs Frequency Bands Plot
**Worksheet Name:** `Signal_Minus_Noise_Frequency_Bands_Plot`
**Function:** `create_signal_minus_noise_frequency_bands_plot()`

- **Purpose:** Shows signal minus noise across frequency bands for all measurements
- **Noise Reference:** Folder-specific noise floor (all NoLeak data from each folder)
- **Calculation:** `average_signal_per_band - average_noise_per_band`
- **Grouping:** By measurement type and folder

### 2. File-Specific Signal Minus Noise vs Frequency Bands Plot
**Worksheet Name:** `File_Specific_Signal_Minus_Noise_Plot`
**Function:** `create_file_specific_signal_minus_noise_plot()`

- **Purpose:** Shows signal minus noise using file-specific noise floors
- **Noise Reference:** File-specific noise floor (only NoLeak data from same base file)
- **Calculation:** `average_signal_per_band - average_noise_per_band`
- **Grouping:** By measurement type, folder, and base filename

## Key Features

### **Signal Minus Noise Calculation**
```python
# For each frequency band:
band_signal = signal[band_mask]
band_noise = noise_floor[band_mask]
signal_minus_noise = np.mean(band_signal) - np.mean(band_noise)

# Ensure non-negative values
signal_minus_noise = max(0, signal_minus_noise)
```

### **Frequency Bands** (Same as SNR analysis)
- **Ultra-Low to Low Acoustic Transition:** 1-100 Hz (10 bands)
- **Low Structural:** 100-500 Hz (8 bands)
- **Mid Frequency:** 500-2000 Hz (2 bands)
- **High Frequency:** 2000-8000 Hz (1 band)
- **Ultrasonic:** 8000-20000 Hz (1 band)

### **Data Handling**
- **Non-negative enforcement:** Signal can't be below noise floor
- **Precision:** Values rounded to 6 decimal places
- **Zero handling:** Missing data shows as 0
- **Averaging:** Multiple measurements per key are averaged

## Chart Configuration

### **Visual Design**
- **Chart Type:** Line chart with markers
- **Colors:** 16 distinguishable colors (same as existing plots)
- **Markers:** Circles, size 5
- **Line Width:** 2 pixels
- **Size:** 1400×700 pixels

### **Axes**
- **X-Axis:** Frequency Bands (text categories)
- **Y-Axis:** Signal Minus Noise (Amplitude Units)
- **Title:** Descriptive based on plot type

### **Legend and Layout**
- **Legend:** Right side with size 12 font
- **Plot Area:** 75% width, 70% height
- **Margins:** 15% left, 15% top

## Implementation Details

### **Integration Points**
```python
# Added to summary comparison creation:
create_signal_minus_noise_frequency_bands_plot(summaryComparisonWorkbook, all_analysis_data)
create_file_specific_signal_minus_noise_plot(summaryComparisonWorkbook, all_analysis_data)
```

### **Data Structure**
- **Global Plot:** Uses folder-level noise floors
- **File-Specific Plot:** Uses individual file noise floors
- **Measurement Keys:** Include distance, folder, and filename information

### **Noise Floor Handling**
- **No epsilon protection needed** (subtraction vs division)
- **Folder-specific:** Average of all NoLeak measurements in folder
- **File-specific:** Average of NoLeak measurements from same base file

## Benefits Over SNR Plots

### **1. Intuitive Units**
- **SNR:** Dimensionless ratios (linear or dB)
- **Signal Minus Noise:** Amplitude units (same as input data)

### **2. Linear Relationship**
- **SNR:** Logarithmic/ratio relationship
- **Signal Minus Noise:** Direct linear difference

### **3. Absolute Magnitude**
- **SNR:** Relative comparison
- **Signal Minus Noise:** Absolute signal strength above noise

### **4. Easier Interpretation**
- **SNR:** Requires understanding of ratios/dB
- **Signal Minus Noise:** Direct "how much signal above noise"

## Use Cases

### **Leak Detection Analysis**
- **Threshold Setting:** Establish minimum signal-above-noise requirements
- **Sensitivity Analysis:** Compare absolute signal strengths
- **Distance Effects:** Observe signal degradation with distance

### **Sensor Performance**
- **Dynamic Range:** Understand sensor's effective range
- **Noise Floor Analysis:** Evaluate baseline performance
- **Signal Strength:** Compare raw signal magnitudes

### **Environmental Assessment**
- **Background Noise:** Assess environmental noise impact
- **Signal Clarity:** Determine signal visibility above ambient
- **Frequency Response:** Analyze frequency-dependent performance

## Data Interpretation

### **High Values**
- Strong signal well above noise floor
- Good leak detection conditions
- Clear frequency band separation

### **Low Values**
- Signal close to noise floor
- Challenging detection conditions
- May require sensitivity adjustments

### **Zero Values**
- Signal at or below noise floor
- No detectable signal in that band
- Possible measurement issues

## Comparison with SNR

| Aspect | SNR | Signal Minus Noise |
|--------|-----|-------------------|
| **Units** | Dimensionless | Amplitude units |
| **Scale** | Logarithmic/Ratio | Linear |
| **Interpretation** | Relative strength | Absolute difference |
| **Zero Handling** | Division by zero issues | Natural subtraction |
| **Extreme Values** | Can approach infinity | Bounded by signal strength |
| **Intuition** | Requires ratio understanding | Direct magnitude |

## Technical Notes

### **Performance**
- **Computational Cost:** Lower than SNR (no division/log operations)
- **Memory Usage:** Similar to existing plots
- **Precision:** 6 decimal places for adequate resolution

### **Robustness**
- **No division by zero concerns**
- **No logarithm domain issues**
- **Natural handling of negative differences**

### **Future Enhancements**
- **Confidence intervals** for signal minus noise values
- **Statistical significance testing** for differences
- **Adaptive thresholding** based on noise characteristics
- **Time-series analysis** of signal minus noise evolution

## Validation

The signal minus noise values should:
- **Be non-negative** (enforced in implementation)
- **Correlate with SNR trends** (higher SNR → higher signal minus noise)
- **Show realistic magnitudes** relative to input data
- **Vary appropriately with distance** (decreasing with distance)