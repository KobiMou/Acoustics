# Acoustics Analysis System

A comprehensive acoustic data analysis system designed for leak detection and signal processing. This system provides advanced FFT analysis, leak detection algorithms, data validation, and batch processing capabilities.

## System Overview

This acoustics analysis system consists of several integrated components:

1. **Core FFT Analysis** (`FFT_sound_txt`) - Advanced FFT processing with leak detection
2. **Data Validation** (`acoustics_data_validator.py`) - Comprehensive data quality assessment
3. **Batch Processing** (`acoustics_batch_processor.py`) - Automated batch analysis workflows
4. **Configuration Management** (`acoustics_config.json`) - Centralized configuration system

## Features

### Core Analysis Features
- **Advanced FFT Processing**: Hanning windowing, configurable sample sizes (131K samples), multiple averages
- **Power Spectral Density (PSD)**: Proper windowing correction factors for accurate power calculations
- **Signal-to-Noise Ratio (SNR)**: Both global and distance-specific SNR calculations
- **Comprehensive Visualization**: Multiple chart types (FFT, Log-Log scale, PSD, SNR plots)

### Leak Detection Capabilities
- **Statistical Analysis**: Confidence-based thresholding with customizable parameters
- **Frequency Band Analysis**: 22 predefined frequency bands covering 1Hz to 20kHz
- **Power Ratio Detection**: dB-based power increase detection
- **Distance-Specific Analysis**: Baseline comparisons specific to measurement distances
- **Composite Scoring**: Weighted scoring system combining multiple detection methods

### Data Quality Assessment
- **Comprehensive Validation**: SNR estimation, outlier detection, sampling consistency
- **Quality Scoring**: 0-100 data integrity scores with detailed recommendations
- **Multiple Detection Methods**: IQR, Z-score, and Modified Z-score outlier detection
- **Frequency Content Analysis**: Spectral peaks, band distribution, and flatness metrics

### Batch Processing
- **Multi-Job Management**: Queue-based job processing with priority support
- **Automatic Organization**: Directory-based job creation and organization
- **Error Recovery**: Configurable retry mechanisms with failure tracking
- **Comprehensive Reporting**: JSON and CSV reports with detailed job summaries

## Installation

### Prerequisites
```bash
pip install numpy pandas matplotlib scipy xlsxwriter
```

### Optional Dependencies
```bash
pip install seaborn  # For enhanced data validation visualizations
```

## Quick Start

### 1. Basic Analysis
```bash
python FFT_sound_txt
```
This runs the core analysis on the configured data directory and generates:
- `summary.xlsx` with all analysis results
- Multiple visualization charts
- Leak detection reports

### 2. Data Validation
```bash
python acoustics_data_validator.py --data-path /path/to/data --output-dir validation_results
```
Generates:
- `validation_results.csv` - Detailed quality metrics
- `validation_report.html` - Comprehensive HTML report

### 3. Batch Processing
```bash
# Create batch configuration
python acoustics_batch_processor.py --create-config

# Run batch processing
python acoustics_batch_processor.py --data-path /path/to/data --job-type both --auto-organize
```

## Configuration

### Main Configuration (`acoustics_config.json`)
```json
{
  "analysis_parameters": {
    "n_samples": 131072,
    "n_averages": 7,
    "skip_rows": 1,
    "hanning_window": true,
    "hanning_correction_factor": 2.67
  },
  "leak_detection": {
    "statistical": {
      "confidence_factor": 3,
      "detection_threshold_multiplier": 1.0
    },
    "power_ratio": {
      "threshold_db": 8,
      "minimum_increase_db": 5
    },
    "scoring_weights": {
      "statistical": 0.4,
      "frequency_bands": 0.4,
      "power_ratio": 0.2
    }
  }
}
```

### Batch Configuration (`batch_config.json`)
```json
{
  "max_concurrent_jobs": 2,
  "retry_failed_jobs": true,
  "max_retries": 3,
  "timeout_minutes": 60
}
```

## Data Format

### Input Data Requirements
- **File Formats**: CSV, TXT, or DAT files
- **Columns**: Minimum 2 columns (time, amplitude)
- **Delimiters**: Automatic detection of whitespace, comma, tab, or semicolon
- **Naming Convention**: Files containing "NoLeak" are used as baseline references
- **Distance Extraction**: Numeric values before "m" in filename (e.g., "25m_measurement.csv")

### Example Data Structure
```
project_data/
├── 10m_NoLeak_baseline.csv
├── 10m_test_measurement1.csv
├── 25m_NoLeak_baseline.csv
├── 25m_test_measurement1.csv
└── 50m_test_measurement1.csv
```

## Output Files

### Core Analysis Output
- **`summary.xlsx`**: Master Excel file with all results
  - Individual measurement sheets
  - Multiple visualization charts
  - Leak detection analysis
  - Distance-specific results

### Chart Types Generated
1. **FFT Plot**: Frequency vs Minimum Absolute Values
2. **Log-Log Plot**: Both axes on logarithmic scale
3. **PSD Plot**: Power Spectral Density analysis
4. **SNR Plot**: Signal-to-Noise Ratio analysis
5. **Distance-Specific SNR**: Distance-based baseline comparisons

### Leak Detection Results
- **Global Analysis**: Using all NoLeak measurements as baseline
- **Distance-Specific Analysis**: Distance-matched baseline comparisons
- **Frequency Band Details**: Analysis across 22 frequency bands
- **Composite Scores**: Overall leak probability ratings (HIGH/MEDIUM/LOW)

## Advanced Usage

### Custom Frequency Bands
Modify the frequency bands in `acoustics_config.json`:
```json
"frequency_bands": [
  {"name": "custom_band", "freq_range": [100, 500], "description": "Custom frequency range"},
  ...
]
```

### Batch Processing Workflow
1. **Setup**: Create configuration files
2. **Organize**: Structure data in directories
3. **Validate**: Run data quality assessment
4. **Analyze**: Execute batch analysis
5. **Review**: Examine reports and results

### Data Quality Validation Workflow
```bash
# Validate data quality first
python acoustics_data_validator.py --data-path data/ --output-dir validation/

# Review validation report
open validation/validation_report.html

# Fix any data quality issues identified

# Proceed with analysis
python FFT_sound_txt
```

## Leak Detection Algorithm

### Multi-Method Approach
1. **Statistical Thresholding**: Mean + N×Standard Deviation
2. **Frequency Band Analysis**: 22 predefined bands with Z-score analysis
3. **Power Ratio Detection**: dB-based power increase detection

### Scoring System
- **Statistical Score**: Detection ratio and exceedance metrics
- **Frequency Band Score**: Percentage of bands showing leak signatures
- **Power Ratio Score**: Maximum power increase analysis
- **Composite Score**: Weighted combination (40% + 40% + 20%)

### Leak Probability Classification
- **HIGH (≥70)**: Strong leak indicators across multiple methods
- **MEDIUM (40-69)**: Moderate leak indicators
- **LOW (<40)**: Minimal or no leak indicators

## Performance Optimization

### Memory Management
- **Sample Size**: 131,072 samples (2^17) for optimal FFT performance
- **Averaging**: 7 averages to reduce noise while maintaining speed
- **Windowing**: Hanning window with proper correction factors

### Processing Speed
- **Batch Processing**: Configurable concurrent job limits
- **Efficient Algorithms**: Optimized FFT and statistical calculations
- **Memory Cleanup**: Automatic temporary file cleanup

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `n_samples` in configuration
   - Decrease `n_averages` for large files
   - Process files individually rather than in batch

2. **File Loading Errors**
   - Check file format and delimiters
   - Verify column structure (minimum 2 columns)
   - Ensure proper file permissions

3. **Poor Leak Detection Performance**
   - Verify NoLeak baseline files are present
   - Check distance matching between baselines and test measurements
   - Adjust detection thresholds in configuration

4. **Slow Processing**
   - Reduce concurrent jobs in batch processing
   - Optimize file I/O by using local storage
   - Consider reducing sample size for preliminary analysis

### Log Files
- Analysis logs: Console output and error messages
- Batch processing logs: `batch_logs/` directory
- Validation logs: Included in validation reports

## Technical Specifications

### Signal Processing
- **FFT Implementation**: NumPy FFT with Hanning windowing
- **Sampling**: Variable rate detection from time stamps
- **Frequency Range**: DC to Nyquist frequency
- **Resolution**: Configurable sample size for frequency resolution control

### Statistical Methods
- **Outlier Detection**: IQR, Z-score, Modified Z-score methods
- **SNR Calculation**: Power-based signal-to-noise estimation
- **Confidence Intervals**: Configurable statistical thresholds

### File I/O
- **Formats Supported**: CSV, TXT, DAT
- **Encoding**: UTF-8 with automatic delimiter detection
- **Output Formats**: Excel (XLSX), CSV, JSON, HTML reports

## API Reference

### Core Functions

#### FFT Analysis
- `perform_fft_analysis()`: Main FFT processing with windowing
- `calculate_psd()`: Power spectral density calculation
- `calculate_snr()`: Signal-to-noise ratio computation

#### Leak Detection
- `detect_leak_statistical()`: Statistical threshold detection
- `detect_leak_frequency_bands()`: Frequency band analysis
- `detect_leak_power_ratio()`: Power ratio detection
- `calculate_leak_detection_score()`: Composite scoring

#### Data Validation
- `validate_file()`: Single file quality assessment
- `validate_directory()`: Batch file validation
- `generate_recommendations()`: Quality improvement suggestions

## Contributing

### Development Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest tests/`
4. Follow PEP 8 coding standards

### Adding New Features
1. Update configuration schema if needed
2. Implement new functionality with proper error handling
3. Add comprehensive logging
4. Update documentation and examples

## License

This acoustics analysis system is proprietary software developed for Arad Technologies Ltd.

## Support

For technical support and questions:
- Review this documentation thoroughly
- Check log files for error details
- Verify data format and configuration
- Contact the development team for advanced troubleshooting

---

*Last updated: 2025-01-21*
*Version: 2.0*