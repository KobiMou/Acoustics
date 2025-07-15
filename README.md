# Audio Analysis for Leak Detection

## Overview
This code processes WAV audio files to analyze acoustic data for leak detection. It extracts specific segments from each WAV file and performs FFT analysis to identify patterns that may indicate leaks.

## Requirements
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Directory Structure
The code expects the following directory structure:
```
audio_data/
├── folder1/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
├── folder2/
│   ├── file3.wav
│   ├── file4.wav
│   └── ...
└── ...
```

## Audio File Processing
For each WAV file, the code extracts two segments:

1. **Leak Segment**: 35 seconds starting 10 seconds from the beginning
2. **NoLeak Segment**: 35 seconds ending 10 seconds before the end

### Minimum File Duration
Each WAV file must be at least 55 seconds long (35 + 10 + 10 seconds) to accommodate both segments with proper offsets.

## Configuration
Update the `path` variable in `main_11.py` to point to your audio data directory:
```python
path = r'D:\OneDrive - Arad Technologies Ltd\ARAD_Projects\ALD\tests\audio_data'
```

## Analysis Parameters
- **n_samples**: 131072 (2^17) - Number of samples per FFT window
- **n_AVG**: 7 - Number of averages for FFT analysis
- **Segment Duration**: 35 seconds for both leak and noleak segments
- **Offset Duration**: 10 seconds from start/end of file

## Output
The code generates:
- `audio_analysis_summary.xlsx` - Excel file containing:
  - Individual worksheets for each extracted segment
  - FFT analysis results (frequency, amplitude, phase, PSD, SNR)
  - Charts showing frequency analysis (linear and log scales)
  - Leak detection analysis results

## Features
- **Automatic Segment Extraction**: Processes both leak and noleak segments from each WAV file
- **FFT Analysis**: Applies Hanning window and calculates Power Spectral Density
- **SNR Calculation**: Signal-to-Noise Ratio analysis
- **Leak Detection**: Multiple algorithms for detecting acoustic leaks
- **Visualization**: Automatic chart generation in Excel format
- **Error Handling**: Robust processing with detailed error messages

## Usage
1. Place your WAV files in the appropriate folder structure
2. Update the `path` variable to point to your data directory
3. Run the script:
   ```bash
   python main_11.py
   ```

The script will process all subfolders and generate a comprehensive analysis report.