import os
import numpy as np
import pandas as pd
import xlsxwriter
import scipy.fftpack
import matplotlib.pyplot as plt
import re
from scipy import stats
import librosa
import soundfile as sf
from scipy.io import wavfile

# Update path to point to folder containing folders with WAV files
path = r'D:\OneDrive - Arad Technologies Ltd\ARAD_Projects\ALD\tests\test_20_07_2025_charge_AW\WAV'

# Get all subfolders in the main path
subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

ListFileNames = []
ListDataFrames = []

n_samples = 131072  # 2^n  -> 131072 (2^17)
n_AVG = 7

# SNR Calculation Configuration
SNR_CONFIG = {
    'noise_floor_method': 'percentile',  # 'mean', 'median', 'percentile', 'trimmed_mean', 'robust_avg'
    'noise_floor_percentile': 10,       # Percentile to use when method='percentile'
    'min_noise_threshold': 1e-12,       # Minimum noise floor to prevent division by very small numbers
    'snr_calculation_method': 'clipped_linear',  # 'linear', 'db', 'clipped_linear', 'clipped_db'
    'max_snr_db': 60,                   # Maximum SNR in dB to clip unrealistic values
    'min_snr_db': -40,                  # Minimum SNR in dB to clip unrealistic values
    'file_specific_method': 'median',   # Noise floor method for file-specific calculations (smaller sample sizes)
    'enable_snr_validation': True,      # Enable SNR validation warnings
    'outlier_detection_method': 'iqr',  # 'iqr', 'zscore', 'percentile'
    'outlier_threshold_factor': 3.0     # Factor for outlier detection
}

# Create list of tuples (filename, dataframe) for sorting
file_data_pairs = []

def safe_log10(x, default=-100):
    """
    Safely compute log10, handling zero/negative values and NaN/INF
    
    Args:
        x: Input value or array
        default: Default value to use for invalid inputs
    
    Returns:
        log10(x) with NaN/INF values replaced by default
    """
    try:
        # Handle arrays
        if hasattr(x, '__len__'):
            result = np.full_like(x, default, dtype=float)
            valid_mask = (x > 0) & np.isfinite(x)
            if np.any(valid_mask):
                result[valid_mask] = np.log10(x[valid_mask])
            return result
        else:
            # Handle scalar
            if x > 0 and np.isfinite(x):
                return np.log10(x)
            else:
                return default
    except:
        return default

def improved_noise_floor_estimation(noise_data_list, method='percentile', percentile=10, min_noise_threshold=1e-12):
    """
    Estimate noise floor using various robust methods
    
    Args:
        noise_data_list: List of noise measurement arrays
        method: 'mean', 'median', 'percentile', 'trimmed_mean', 'robust_avg'
        percentile: Percentile to use for percentile method (default 10th percentile)
        min_noise_threshold: Minimum noise floor to prevent division by very small numbers
    
    Returns:
        Robust noise floor estimate
    """
    if not noise_data_list:
        return None
    
    # Stack all noise measurements
    noise_stack = np.vstack(noise_data_list)
    
    if method == 'mean':
        noise_floor = np.mean(noise_stack, axis=0)
    elif method == 'median':
        noise_floor = np.median(noise_stack, axis=0)
    elif method == 'percentile':
        noise_floor = np.percentile(noise_stack, percentile, axis=0)
    elif method == 'trimmed_mean':
        # Remove top and bottom 10% and take mean
        sorted_noise = np.sort(noise_stack, axis=0)
        trim_size = max(1, len(noise_data_list) // 10)
        if len(noise_data_list) > 2 * trim_size:
            trimmed_noise = sorted_noise[trim_size:-trim_size, :]
            noise_floor = np.mean(trimmed_noise, axis=0)
        else:
            noise_floor = np.median(sorted_noise, axis=0)
    elif method == 'robust_avg':
        # Use median + small offset from minimum as robust estimator
        median_noise = np.median(noise_stack, axis=0)
        min_noise = np.min(noise_stack, axis=0)
        noise_floor = median_noise + 0.1 * (median_noise - min_noise)
    else:
        noise_floor = np.mean(noise_stack, axis=0)
    
    # Apply minimum threshold to prevent division by very small numbers
    noise_floor = np.maximum(noise_floor, min_noise_threshold)
    
    return noise_floor

def calculate_robust_snr(signal, noise_floor, method='linear', max_snr_db=60, min_snr_db=-40):
    """
    Calculate robust SNR with multiple options and safeguards
    
    Args:
        signal: Signal power/amplitude array
        noise_floor: Noise floor estimate
        method: 'linear', 'db', 'log10', 'clipped_linear', 'clipped_db'
        max_snr_db: Maximum SNR in dB to prevent unrealistic values
        min_snr_db: Minimum SNR in dB to prevent unrealistic values
    
    Returns:
        Robust SNR estimate
    """
    # Ensure inputs are valid
    signal = np.asarray(signal)
    noise_floor = np.asarray(noise_floor)
    
    # Basic linear SNR calculation
    snr_linear = signal / noise_floor
    
    if method == 'linear':
        return snr_linear
    
    elif method == 'db':
        # Convert to dB with clipping
        snr_db = 10 * safe_log10(snr_linear, default=min_snr_db/10)
        return snr_db
    
    elif method == 'log10':
        # Log10 scale
        return safe_log10(snr_linear, default=min_snr_db/10)
    
    elif method == 'clipped_linear':
        # Clip linear SNR to reasonable range
        max_snr_linear = 10**(max_snr_db/10)
        min_snr_linear = 10**(min_snr_db/10)
        return np.clip(snr_linear, min_snr_linear, max_snr_linear)
    
    elif method == 'clipped_db':
        # Convert to dB and clip
        snr_db = 10 * safe_log10(snr_linear, default=min_snr_db)
        return np.clip(snr_db, min_snr_db, max_snr_db)
    
    else:
        return snr_linear

def detect_snr_outliers(snr_values, method='iqr', threshold_factor=3.0):
    """
    Detect and flag SNR outliers that might indicate calculation issues
    
    Args:
        snr_values: Array of SNR values
        method: 'iqr', 'zscore', 'percentile'
        threshold_factor: Factor for outlier detection threshold
    
    Returns:
        Boolean mask indicating outliers
    """
    snr_array = np.asarray(snr_values)
    
    if method == 'iqr':
        # Interquartile range method
        q75, q25 = np.percentile(snr_array, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - threshold_factor * iqr
        upper_bound = q75 + threshold_factor * iqr
        outliers = (snr_array < lower_bound) | (snr_array > upper_bound)
    
    elif method == 'zscore':
        # Z-score method
        mean_snr = np.mean(snr_array)
        std_snr = np.std(snr_array)
        z_scores = np.abs((snr_array - mean_snr) / std_snr)
        outliers = z_scores > threshold_factor
    
    elif method == 'percentile':
        # Percentile-based method
        lower_percentile = threshold_factor
        upper_percentile = 100 - threshold_factor
        lower_bound, upper_bound = np.percentile(snr_array, [lower_percentile, upper_percentile])
        outliers = (snr_array < lower_bound) | (snr_array > upper_bound)
    
    else:
        outliers = np.zeros_like(snr_array, dtype=bool)
    
    return outliers

def validate_snr_calculation(signal, noise_floor, snr_result, freq_array=None):
    """
    Validate SNR calculation and provide diagnostic information
    
    Args:
        signal: Original signal data
        noise_floor: Noise floor used
        snr_result: Calculated SNR
        freq_array: Frequency array for reporting
    
    Returns:
        Dictionary with validation results and recommendations
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'recommendations': [],
        'statistics': {}
    }
    
    # Check for very low noise floor values
    very_low_noise = noise_floor < 1e-10
    if np.any(very_low_noise):
        count = np.sum(very_low_noise)
        validation['warnings'].append(f"Very low noise floor detected at {count} frequency bins (< 1e-10)")
        validation['recommendations'].append("Consider using a higher minimum noise threshold")
    
    # Check for unrealistically high SNR values
    if hasattr(snr_result, '__len__'):
        high_snr = snr_result > 1000  # > 30 dB in linear scale
        if np.any(high_snr):
            count = np.sum(high_snr)
            max_snr = np.max(snr_result[~np.isinf(snr_result)])
            validation['warnings'].append(f"Very high SNR values detected at {count} frequency bins (max: {max_snr:.1f})")
            validation['recommendations'].append("Consider using clipped SNR calculation method")
    
    # Check for infinite or NaN values
    invalid_values = ~np.isfinite(snr_result)
    if np.any(invalid_values):
        count = np.sum(invalid_values)
        validation['warnings'].append(f"Non-finite SNR values detected at {count} frequency bins")
        validation['is_valid'] = False
    
    # Calculate statistics
    finite_snr = snr_result[np.isfinite(snr_result)]
    if len(finite_snr) > 0:
        validation['statistics'] = {
            'mean_snr': np.mean(finite_snr),
            'median_snr': np.median(finite_snr),
            'std_snr': np.std(finite_snr),
            'min_snr': np.min(finite_snr),
            'max_snr': np.max(finite_snr),
            'valid_points': len(finite_snr),
            'total_points': len(snr_result)
        }
    
    return validation

def safe_divide(numerator, denominator, default=0):
    """
    Safely divide two values, handling division by zero and NaN/INF
    
    Args:
        numerator: Numerator value or array
        denominator: Denominator value or array
        default: Default value to use for invalid divisions
    
    Returns:
        numerator/denominator with NaN/INF values replaced by default
    """
    try:
        # Handle division by zero and invalid values
        if hasattr(denominator, '__len__'):
            # Both numerator and denominator are arrays
            result = np.full_like(numerator, default, dtype=float)
            valid_mask = (denominator != 0) & np.isfinite(denominator) & np.isfinite(numerator)
            if np.any(valid_mask):
                result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
            return result
        else:
            # Denominator is scalar
            if denominator != 0 and np.isfinite(denominator):
                if hasattr(numerator, '__len__'):
                    # Numerator is array, denominator is scalar
                    result = np.full_like(numerator, default, dtype=float)
                    valid_mask = np.isfinite(numerator)
                    if np.any(valid_mask):
                        result[valid_mask] = numerator[valid_mask] / denominator
                    return result
                else:
                    # Both are scalars
                    if np.isfinite(numerator):
                        return numerator / denominator
                    else:
                        return default
            else:
                # Denominator is zero or invalid
                if hasattr(numerator, '__len__'):
                    return np.full_like(numerator, default, dtype=float)
                else:
                    return default
    except:
        if hasattr(numerator, '__len__'):
            return np.full_like(numerator, default, dtype=float)
        else:
            return default

def sanitize_excel_value(value, default=0):
    """
    Sanitize a value before writing to Excel, handling NaN and INF values
    
    Args:
        value: Value to sanitize
        default: Default value to use for invalid values
    
    Returns:
        Sanitized value safe for Excel
    """
    try:
        if np.isnan(value) or np.isinf(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_audio_segments(wav_file_path, segment_type='leak'):
    """
    Extract audio segments from WAV file
    - Leak: 35 seconds starting after 10 seconds from beginning
    - NoLeak: 35 seconds at the end with 10 seconds offset from end
    """
    try:
        # Load audio file
        audio_data, sample_rate = librosa.load(wav_file_path, sr=None)
        
        # Calculate segment parameters
        segment_duration = 35  # seconds
        offset_duration = 10   # seconds
        
        segment_samples = int(segment_duration * sample_rate)
        offset_samples = int(offset_duration * sample_rate)
        
        # Check if file is long enough for both segments
        min_required_duration = segment_duration + 2 * offset_duration  # 55 seconds minimum
        file_duration = len(audio_data) / sample_rate
        
        if file_duration < min_required_duration:
            print(f"Warning: {wav_file_path} is only {file_duration:.1f}s long, need at least {min_required_duration}s")
            return None, None
        
        if segment_type.lower() == 'leak':
            # Extract leak segment: start after 10 seconds, duration 35 seconds
            start_sample = offset_samples
            end_sample = start_sample + segment_samples
        else:  # noleak
            # Extract noleak segment: 35 seconds at end with 10 seconds offset
            end_sample = len(audio_data) - offset_samples
            start_sample = end_sample - segment_samples
        
        # Double-check bounds
        if start_sample < 0 or end_sample > len(audio_data):
            print(f"Warning: Invalid segment bounds for {wav_file_path} ({segment_type})")
            return None, None
        
        # Extract the segment
        segment_data = audio_data[start_sample:end_sample]
        
        # Create time array starting from 0 for each segment
        time_array = np.arange(len(segment_data)) / sample_rate
        
        print(f"Extracted {segment_type} segment from {os.path.basename(wav_file_path)}: {len(segment_data)} samples @ {sample_rate} Hz")
        
        return time_array, segment_data
        
    except Exception as e:
        print(f"Error processing {wav_file_path}: {e}")
        return None, None

def process_wav_files_in_folder(folder_path):
    """Process all WAV files in a folder and extract leak/noleak segments"""
    all_wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    
    # Filter WAV files to only include those with distance pattern (number + "m")
    wav_files = []
    for f in all_wav_files:
        if re.search(r'\d+m', f):
            wav_files.append(f)
        else:
            print(f"  Skipping {f} - no distance pattern found (expecting format like '5m', '10m', etc.)")
    
    print(f"  Found {len(wav_files)} WAV files with distance pattern (out of {len(all_wav_files)} total)")
    
    processed_data = []
    
    for wav_file in wav_files:
        wav_path = os.path.join(folder_path, wav_file)
        base_name = os.path.splitext(wav_file)[0]
        
        # Extract leak segment
        time_leak, signal_leak = extract_audio_segments(wav_path, 'leak')
        if time_leak is not None and signal_leak is not None:
            # Create DataFrame similar to original text format
            leak_df = pd.DataFrame({
                0: time_leak,
                1: signal_leak
            })
            processed_data.append((f"{base_name}_leak", leak_df))
        
        # Extract noleak segment
        time_noleak, signal_noleak = extract_audio_segments(wav_path, 'noleak')
        if time_noleak is not None and signal_noleak is not None:
            # Create DataFrame similar to original text format
            noleak_df = pd.DataFrame({
                0: time_noleak,
                1: signal_noleak
            })
            processed_data.append((f"{base_name}_noleak", noleak_df))
    
    return processed_data

# Process all subfolders
print(f"Starting audio analysis from: {path}")
print(f"Found {len(subfolders)} subfolders to process")
print(f"Note: Only processing WAV files with distance pattern (e.g., 'sensor_5m.wav', 'test_10m.wav')")
print(f"      Additional words after distance are included in worksheet names:")
print(f"      - 'sensor_5m_leak' -> '5m_Leak'")
print(f"      - 'sensor_5m_front_leak' -> '5m_front_Leak'")
print(f"      - 'test_10m_sensor_noleak' -> '10m_sensor_NoLeak'")
print(f"      Files without distance pattern (e.g., 'test.wav', 'background.wav') will be ignored")

total_wav_files = 0
total_segments = 0

def process_folder_analysis(subfolder_path, subfolder_name, folder_data):
    """Process and analyze data for a single folder"""
    
    if not folder_data:
        print(f"No data to process for folder: {subfolder_name}")
        return
    
    print(f"Creating analysis for folder: {subfolder_name}")
    
    # Create separate workbook for this folder
    summary_filename = os.path.join(subfolder_path, f"{subfolder_name}_analysis_summary.xlsx")
    summaryWorkbook = xlsxwriter.Workbook(summary_filename, {'nan_inf_to_errors': True})
    
    # Track used worksheet names to ensure uniqueness
    used_worksheet_names = set()
    
    # Store chart series info and analysis data for this folder
    chart_series_info = []
    analysis_data = []
    
    # Convert folder data to the format expected by the analysis
    ListFileNames = []
    ListDataFrames = []
    
    # Sort folder data by filename
    folder_data.sort(key=lambda x: extract_number_before_m(x[0]))
    
    for fName, DataFrame in folder_data:
        ListFileNames.append(fName)
        ListDataFrames.append(DataFrame)
    
    iLoop = 0
    
    for DataFrame in ListDataFrames:
        DataFrameSize = len(DataFrame)
        
        if n_AVG <= DataFrameSize//n_samples:
            time_Array = DataFrame[0][0:n_samples].to_numpy()
            
            listSampleRate = []
            Array_list = []
            fft = []
            fftAbs = []
            
            N = len(time_Array)
            SampleRateArray = None
            
            for i in range(1, N):
                sampleRate = time_Array[i] - time_Array[i-1]
                listSampleRate.append(sampleRate)

            st = sum(listSampleRate) / len(listSampleRate)
            st = round(st, 9)
            fs = 1 / st        
            
            fftFreq = np.fft.fftfreq(N, st)
            
            # Create Hanning window
            hanning_window = np.hanning(N)
            
            for j in range(n_AVG):
                data_Array = DataFrame[1][j*n_samples:(j+1)*n_samples].to_numpy()
                data_Array = np.transpose(data_Array)

                Array_list.append(data_Array)

                # Apply Hanning window to reduce spectral leakage
                windowed_data = data_Array * hanning_window
                
                # Calculate FFT for this segment
                fft.append(np.fft.fft(windowed_data))
            
            # Calculate fftAbs after all FFTs are computed
            fftAbs = np.abs(fft)/N*2*2 # Normalize result for correct amplitude (×2 for Hanning window compensation)
        
        fft_AVG = np.mean(fft, axis=0)  
        fftAbs_AVG = np.mean(fftAbs, axis=0)    
        fft_MIN = np.min(fft, axis=0)  
        fftAbs_MIN = np.min(fftAbs, axis=0)
        
        # Calculate PSD (Power Spectral Density) - using improved method from reference code
        # Hanning window correction factor for PSD (compensate for power loss)
        hanning_correction = 8/3  # Correction factor for Hanning window power
        psd = []
        for j in range(n_AVG):
            psd_single = (np.abs(fft[j])**2) / (N * fs) * hanning_correction
            psd.append(psd_single)
        
        psd_AVG = np.mean(psd, axis=0)
        
        # Calculate SNR (will be updated after all files are processed)
        snr = np.zeros_like(psd_AVG)  # Placeholder, will be calculated later
        snr_file = np.zeros_like(psd_AVG)  # Placeholder for file-specific SNR
        snr_fft = np.zeros_like(psd_AVG)  # Placeholder for FFT-based SNR
        snr_fft_file = np.zeros_like(psd_AVG)  # Placeholder for file-specific FFT-based SNR
        
        # Create worksheet name based on distance and leak/noleak designation
        filename = ListFileNames[iLoop]
        
        # Extract distance and any additional word after it
        distance_match = re.search(r'(\d+)m(?:[_\s]?([A-Za-z]+))?', filename)
        if distance_match:
            distance_str = f"{distance_match.group(1)}m"
            additional_word = distance_match.group(2)  # The word after 'm' (if any)
            
            # Build distance part with additional word if present
            if additional_word and additional_word.lower() not in ['leak', 'noleak']:
                distance_part = f"{distance_str}_{additional_word}"
            else:
                distance_part = distance_str
                additional_word = None  # No additional word found
        else:
            distance_part = "Unknown"
            distance_str = "Unknown"
            additional_word = None
        
        print(f"    Distance pattern found: '{distance_part}' from filename: {filename}")
        
        # Determine leak/noleak designation
        if '_leak' in filename.lower():
            leak_designation = "Leak"
        elif '_noleak' in filename.lower():
            leak_designation = "NoLeak"
        else:
            leak_designation = "Unknown"
        
        # Create worksheet name: "5m_sensor_Leak", "10m_test_NoLeak", etc.
        worksheet_name = f"{distance_part}_{leak_designation}"
        
        # Ensure uniqueness by adding suffix if needed
        original_worksheet_name = worksheet_name
        counter = 1
        while worksheet_name in used_worksheet_names:
            worksheet_name = f"{original_worksheet_name}_{counter}"
            counter += 1
        
        used_worksheet_names.add(worksheet_name)
        sanitized_sheet_name = worksheet_name
        
        # Log the worksheet name creation
        print(f"    Creating worksheet: '{sanitized_sheet_name}' for file: {os.path.basename(filename)}")
        if additional_word:
            print(f"      Enhanced naming: distance='{distance_str}', additional='{additional_word}'")
        
        # Store chart series info for this file
        chart_series_info.append({
            'filename': ListFileNames[iLoop],
            'sheet_name': sanitized_sheet_name,
            'data_points': N//2
        })
        
        # Store analysis data for leak detection
        analysis_data.append({
            'filename': ListFileNames[iLoop],
            'frequency': fftFreq[:N//2],
            'psd_avg': psd_AVG[:N//2],
            'fft_abs_min': fftAbs_MIN[:N//2],
            'is_noleak': 'noleak' in ListFileNames[iLoop].lower()
        })
        
        # Create worksheet in summary workbook for this file
        summaryWorksheet = summaryWorkbook.add_worksheet(sanitized_sheet_name)
        
        # Write headers for summary worksheet
        headers = ['Time', 'Data', 'Frequency', 'FFT_Real', 'FFT_Imag', 'FFT_Abs', 'FFT_MIN_Real', 'FFT_MIN_Imag', 'FFT_MIN_Abs', 'PSD', 'SNR', 'SNR_File', 'SNR_FFT', 'SNR_FFT_File']
        for col, header in enumerate(headers):
            summaryWorksheet.write(0, col, header)
        
        # Write data to summary worksheet
        for i in range(N//2): # //2 for only positive side plotting 
            summaryWorksheet.write(i+1,0,sanitize_excel_value(time_Array[i]))
            summaryWorksheet.write(i+1,1,sanitize_excel_value(data_Array[i]))
            summaryWorksheet.write(i+1,2,sanitize_excel_value(fftFreq[i]))
            summaryWorksheet.write(i+1,3,sanitize_excel_value(fft_AVG[i].real))
            summaryWorksheet.write(i+1,4,sanitize_excel_value(fft_AVG[i].imag))
            summaryWorksheet.write(i+1,5,sanitize_excel_value(fftAbs_AVG[i]))
            summaryWorksheet.write(i+1,6,sanitize_excel_value(fft_MIN[i].real))
            summaryWorksheet.write(i+1,7,sanitize_excel_value(fft_MIN[i].imag))
            summaryWorksheet.write(i+1,8,sanitize_excel_value(fftAbs_MIN[i]))
            summaryWorksheet.write(i+1,9,sanitize_excel_value(psd_AVG[i]))
            summaryWorksheet.write(i+1,10,sanitize_excel_value(snr[i]))
            summaryWorksheet.write(i+1,11,sanitize_excel_value(snr_file[i]))
            summaryWorksheet.write(i+1,12,sanitize_excel_value(snr_fft[i]))
            summaryWorksheet.write(i+1,13,sanitize_excel_value(snr_fft_file[i]))
            
        iLoop = iLoop + 1
    
    # Leak Detection Functions (improved from reference code)
    def detect_leak_statistical(leak_psd, noleak_baseline, confidence_factor=3):
        """Detect leaks using statistical thresholding"""
        noise_mean = np.mean(noleak_baseline, axis=0)
        noise_std = np.std(noleak_baseline, axis=0)
        detection_threshold = noise_mean + confidence_factor * noise_std
        leak_detection = leak_psd > detection_threshold
        detection_ratio = np.sum(leak_detection) / len(leak_detection)
        max_exceedance = np.max(leak_psd / np.maximum(detection_threshold, np.finfo(float).eps))
        
        return {
            'leak_detected': np.any(leak_detection),
            'detection_mask': leak_detection,
            'detection_ratio': detection_ratio,
            'max_exceedance_ratio': max_exceedance,
            'threshold': detection_threshold
        }

    def detect_leak_frequency_bands(freq, leak_psd, noleak_baseline):
        """Analyze specific frequency bands where leaks typically occur"""
        leak_bands = [
            # Very low frequency bands (1-100 Hz) - 10 bands
            (1, 10),       # Ultra-low frequency structural
            (10, 20),      # Very low frequency vibrations
            (20, 30),      # Low frequency structural response
            (30, 40),      # Mechanical vibrations
            (40, 50),      # Power line and mechanical harmonics
            (50, 60),      # Power frequency range
            (60, 70),      # Post-power frequency
            (70, 80),      # Low acoustic range
            (80, 90),      # Pre-acoustic range
            (90, 100),     # Low acoustic transition
            # Refined frequency bands
            # 100-500 Hz range in 50 Hz increments
            (100, 150),    # Low structural range 1
            (150, 200),    # Low structural range 2
            (200, 250),    # Low structural range 3
            (250, 300),    # Low structural range 4
            (300, 350),    # Low structural range 5
            (350, 400),    # Low structural range 6
            (400, 450),    # Low structural range 7
            (450, 500),    # Low structural range 8
            # 500-2000 Hz range in 1000 Hz jumps
            (500, 1500),   # Mid frequency acoustic emissions 1
            (1500, 2000),  # Mid frequency acoustic emissions 2
            # Original higher frequency bands
            (2000, 8000),  # High frequency turbulence
            (8000, 20000)  # Ultrasonic range
        ]
        
        results = {}
        
        for i, (f_low, f_high) in enumerate(leak_bands):
            band_mask = (freq >= f_low) & (freq <= f_high)
            
            if np.any(band_mask):
                leak_band = leak_psd[band_mask]
                baseline_band = noleak_baseline[:, band_mask]
                
                baseline_mean = np.mean(baseline_band)
                baseline_std = np.std(baseline_band)
                leak_mean = np.mean(leak_band)
                
                snr_ratio = leak_mean / np.maximum(baseline_mean, np.finfo(float).eps)
                z_score = (leak_mean - baseline_mean) / np.maximum(baseline_std, np.finfo(float).eps)
                
                results[f'band_{f_low}_{f_high}Hz'] = {
                    'snr_ratio': snr_ratio,
                    'z_score': z_score,
                    'leak_detected': z_score > 3,
                    'frequency_range': (f_low, f_high)
                }
        
        return results

    def detect_leak_power_ratio(leak_psd, noleak_baseline, threshold_dB=8):
        """Detect based on power ratio in dB"""
        baseline_power = np.mean(noleak_baseline, axis=0)
        baseline_power = np.maximum(baseline_power, np.finfo(float).eps)
        
        power_ratio_dB = 10 * np.log10(leak_psd / baseline_power)
        leak_detected = power_ratio_dB > threshold_dB
        
        return {
            'power_ratio_dB': power_ratio_dB,
            'leak_detected': np.any(leak_detected),
            'detection_mask': leak_detected,
            'max_power_increase_dB': np.max(power_ratio_dB),
            'mean_power_increase_dB': np.mean(power_ratio_dB)
        }

    def calculate_leak_detection_score(freq, leak_psd, noleak_baseline):
        """Comprehensive leak detection scoring"""
        stat_result = detect_leak_statistical(leak_psd, noleak_baseline, confidence_factor=3)
        band_result = detect_leak_frequency_bands(freq, leak_psd, noleak_baseline)
        power_result = detect_leak_power_ratio(leak_psd, noleak_baseline, threshold_dB=8)
        
        # Calculate composite score
        scores = []
        
        # Statistical score (0-100)
        stat_score = min(100, stat_result['detection_ratio'] * 100 + stat_result['max_exceedance_ratio'] * 20)
        scores.append(stat_score)
        
        # Frequency band score
        band_detections = sum(1 for band in band_result.values() if band['leak_detected'])
        band_score = (band_detections / max(1, len(band_result))) * 100
        scores.append(band_score)
        
        # Power ratio score
        power_score = min(100, max(0, power_result['max_power_increase_dB'] - 5) * 10)
        scores.append(power_score)
        
        # Weighted composite score
        composite_score = np.average(scores, weights=[0.4, 0.4, 0.2])
        
        return {
            'composite_score': composite_score,
            'leak_probability': 'HIGH' if composite_score > 70 else 'MEDIUM' if composite_score > 40 else 'LOW',
            'individual_scores': {
                'statistical': stat_score,
                'frequency_bands': band_score, 
                'power_ratio': power_score
            },
            'detailed_results': {
                'statistical': stat_result,
                'frequency_bands': band_result,
                'power_ratio': power_result
            }
        }
    
    def analyze_leak_detection(analysis_data):
        """Analyze all measurements for leak detection"""
        # Separate NoLeak and potential leak data
        noleak_data = []
        leak_data = []
        
        for data in analysis_data:
            if data['is_noleak']:
                noleak_data.append(data['psd_avg'])
            else:
                leak_data.append((data['filename'], data['psd_avg'], data['frequency']))
        
        if not noleak_data:
            return "No NoLeak baseline data found"
        
        # Stack NoLeak data for baseline
        noleak_baseline = np.vstack(noleak_data)
        
        # Analyze each potential leak measurement
        results = {}
        
        for filename, leak_psd, frequency in leak_data:
            detection_result = calculate_leak_detection_score(frequency, leak_psd, noleak_baseline)
            results[filename] = detection_result
        
        return results
    
    # Calculate and update SNR values in worksheets
    if analysis_data:
        # Calculate global noise floor from all NoLeak measurements
        noleak_psd_data = []
        noleak_fft_data = []
        for data in analysis_data:
            if data['is_noleak']:
                noleak_psd_data.append(data['psd_avg'])
                noleak_fft_data.append(data['fft_abs_min'])
        
        if noleak_psd_data:
            # Create improved global noise floor baseline using robust methods
            noise_floor = improved_noise_floor_estimation(
                noleak_psd_data, 
                method=SNR_CONFIG['noise_floor_method'],
                percentile=SNR_CONFIG['noise_floor_percentile'], 
                min_noise_threshold=SNR_CONFIG['min_noise_threshold']
            )
            
            # Create improved global FFT-based noise floor baseline
            fft_noise_floor = improved_noise_floor_estimation(
                noleak_fft_data, 
                method=SNR_CONFIG['noise_floor_method'],
                percentile=SNR_CONFIG['noise_floor_percentile'],
                min_noise_threshold=SNR_CONFIG['min_noise_threshold']
            )
            
            # Update SNR data for all measurements with improved robust calculation
            for data in analysis_data:
                # Calculate robust SNR with clipping to prevent unrealistic values
                snr = calculate_robust_snr(
                    data['psd_avg'], 
                    noise_floor, 
                    method=SNR_CONFIG['snr_calculation_method'],
                    max_snr_db=SNR_CONFIG['max_snr_db'],
                    min_snr_db=SNR_CONFIG['min_snr_db']
                )
                
                # Calculate robust FFT-based SNR with clipping
                snr_fft = calculate_robust_snr(
                    data['fft_abs_min'], 
                    fft_noise_floor,
                    method=SNR_CONFIG['snr_calculation_method'],
                    max_snr_db=SNR_CONFIG['max_snr_db'],
                    min_snr_db=SNR_CONFIG['min_snr_db']
                )
                
                # Validate SNR calculation and print warnings if needed (if enabled)
                if SNR_CONFIG['enable_snr_validation']:
                    validation = validate_snr_calculation(data['psd_avg'], noise_floor, snr)
                    if validation['warnings']:
                        print(f"SNR Validation warnings for {data['filename']}:")
                        for warning in validation['warnings']:
                            print(f"  - {warning}")
                    if validation['recommendations']:
                        for rec in validation['recommendations']:
                            print(f"  Recommendation: {rec}")
                
                # Update the worksheet with SNR values
                sheet_name = data['filename']
                
                # Find the corresponding sanitized sheet name
                sanitized_sheet_name = None
                for series_info in chart_series_info:
                    if series_info['filename'] == sheet_name:
                        sanitized_sheet_name = series_info['sheet_name']
                        break
                
                if sanitized_sheet_name:
                    # Find and update the corresponding worksheet
                    for ws in summaryWorkbook.worksheets():
                        if ws.get_name() == sanitized_sheet_name:
                            # Update SNR column (column K, index 10)
                            for j in range(len(snr)):
                                ws.write(j+1, 10, sanitize_excel_value(snr[j]))
                            # Update FFT-based SNR column (column M, index 12)
                            for j in range(len(snr_fft)):
                                ws.write(j+1, 12, sanitize_excel_value(snr_fft[j]))
                            break

    # Perform leak detection analysis
    leak_detection_results = analyze_leak_detection(analysis_data)
    
    def analyze_leak_detection_distance_specific(analysis_data):
        """Analyze leak detection using file-specific noise floors"""
        
        # Group data by WAV file base name (without _leak/_noleak suffix)
        wav_file_groups = {}
        
        for data in analysis_data:
            # Extract base filename without _leak/_noleak suffix
            base_filename = re.sub(r'_(leak|noleak)$', '', data['filename'])
            
            if base_filename not in wav_file_groups:
                wav_file_groups[base_filename] = {'noleak': [], 'potential_leaks': []}
            
            if data['is_noleak']:
                wav_file_groups[base_filename]['noleak'].append(data)
            else:
                wav_file_groups[base_filename]['potential_leaks'].append(data)
        
        # Analyze each WAV file group
        results = {}
        
        for base_filename, group in wav_file_groups.items():
            # Extract distance from base filename
            distance_match = re.search(r'(\d+)m', base_filename)
            distance = int(distance_match.group(1)) if distance_match else 0
            
            if not group['noleak']:
                # No NoLeak baseline for this WAV file
                for leak_data in group['potential_leaks']:
                    results[leak_data['filename']] = {
                        'error': f'No NoLeak baseline available for WAV file {base_filename}',
                        'distance': distance,
                        'baseline_available': False
                    }
                continue
            
            # Create WAV file-specific baseline (each WAV file uses only its own NoLeak data)
            noleak_baseline = np.vstack([data['psd_avg'] for data in group['noleak']])
            
            # Analyze potential leaks for this WAV file
            for leak_data in group['potential_leaks']:
                detection_result = calculate_leak_detection_score(
                    leak_data['frequency'], leak_data['psd_avg'], noleak_baseline
                )
                
                # Add WAV file-specific information
                detection_result['distance'] = distance
                detection_result['baseline_measurements'] = len(group['noleak'])
                detection_result['baseline_files'] = [data['filename'] for data in group['noleak']]
                detection_result['baseline_available'] = True
                
                results[leak_data['filename']] = detection_result
        
        return results
    
    # Perform distance-specific leak detection analysis
    leak_detection_distance_specific = analyze_leak_detection_distance_specific(analysis_data)
    
    # Create plot worksheet
    if chart_series_info:
        plot_sheet_name = 'Plot'
        counter = 1
        while plot_sheet_name in used_worksheet_names:
            plot_sheet_name = f'Plot_{counter}'
            counter += 1
        used_worksheet_names.add(plot_sheet_name)
        plotWorksheet = summaryWorkbook.add_worksheet(plot_sheet_name)
        
        # Create scatter chart with straight lines
        chart = summaryWorkbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
        
        # Define grey scale colors for NoLeak series (bright and highly distinguishable)
        grey_colors = ['#808080', '#959595', '#AAAAAA', '#BFBFBF', '#D4D4D4', '#E9E9E9', '#F0F0F0', '#F8F8F8']
        
        # Define highly distinguishable colors for regular series
        distinguishable_colors = ['#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF', '#FF0080', '#00FFFF', '#FFFF00', 
                                 '#800000', '#000080', '#008000', '#804000', '#400080', '#800040', '#008080', '#808000']
        
        # First add all non-NoLeak series (colorful lines in background)
        color_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' not in series_info['sheet_name']:
                series_config = {
                    'name': series_info['sheet_name'],
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 8, series_info['data_points'], 8],      # FFT_MIN_Abs column (column I)
                    'line': {'width': 2}
                }
                # Apply distinguishable color for regular series
                regular_color = distinguishable_colors[color_index % len(distinguishable_colors)]
                series_config['line']['color'] = regular_color
                color_index += 1
                chart.add_series(series_config)
        
        # Then add all NoLeak series (grey lines in front)
        grey_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' in series_info['sheet_name']:
                series_config = {
                    'name': series_info['sheet_name'],
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 8, series_info['data_points'], 8],      # FFT_MIN_Abs column (column I)
                    'line': {'width': 2}
                }
                # Apply grey color in ascending order (darker to lighter)
                grey_color = grey_colors[grey_index % len(grey_colors)]
                series_config['line']['color'] = grey_color
                grey_index += 1
                chart.add_series(series_config)
        
        # Configure chart with logarithmic frequency axis
        chart.set_title({'name': 'FFT Frequency vs Minimum Absolute Values', 'name_font': {'size': 12}})
        chart.set_x_axis({
            'name': 'Frequency (Hz)',
            'log_base': 10,
            'label_position': 'low',
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chart.set_y_axis({
            'name': 'FFT Minimum Absolute Values',
            'label_position': 'low',
            'name_layout': {'x': 0.02, 'y': 0.5},
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chart.set_size({'width': 1400, 'height': 700})
        chart.set_plotarea({'layout': {'x': 0.15, 'y': 0.15, 'width': 0.75, 'height': 0.70}})
        chart.set_legend({'font': {'size': 12}})
        
        # Insert chart into worksheet
        plotWorksheet.insert_chart('A1', chart)
        
        # Create second plot worksheet with logarithmic Y-axis
        plot_log_sheet_name = 'Plot_Log_Scale'
        counter = 1
        while plot_log_sheet_name in used_worksheet_names:
            plot_log_sheet_name = f'Plot_Log_Scale_{counter}'
            counter += 1
        used_worksheet_names.add(plot_log_sheet_name)
        plotLogWorksheet = summaryWorkbook.add_worksheet(plot_log_sheet_name)
        
        # Create scatter chart with straight lines (same as first plot)
        chartLog = summaryWorkbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
        
        # First add all non-NoLeak series (colorful lines in background)
        color_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' not in series_info['sheet_name']:
                series_config = {
                    'name': series_info['sheet_name'],
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 8, series_info['data_points'], 8],      # FFT_MIN_Abs column (column I)
                    'line': {'width': 2}
                }
                # Apply distinguishable color for regular series
                regular_color = distinguishable_colors[color_index % len(distinguishable_colors)]
                series_config['line']['color'] = regular_color
                color_index += 1
                chartLog.add_series(series_config)
        
        # Then add all NoLeak series (grey lines in front)
        grey_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' in series_info['sheet_name']:
                series_config = {
                    'name': series_info['sheet_name'],
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 8, series_info['data_points'], 8],      # FFT_MIN_Abs column (column I)
                    'line': {'width': 2}
                }
                # Apply grey color in ascending order (darker to lighter)
                grey_color = grey_colors[grey_index % len(grey_colors)]
                series_config['line']['color'] = grey_color
                grey_index += 1
                chartLog.add_series(series_config)
        
        # Configure chart with logarithmic scales for both axes
        chartLog.set_title({'name': 'FFT Frequency vs Minimum Absolute Values (Log-Log Scale)', 'name_font': {'size': 12}})
        chartLog.set_x_axis({
            'name': 'Frequency (Hz)',
            'log_base': 10,
            'label_position': 'low',
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartLog.set_y_axis({
            'name': 'FFT Minimum Absolute Values',
            'log_base': 10,
            'label_position': 'low',
            'name_layout': {'x': 0.02, 'y': 0.5},
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartLog.set_size({'width': 1400, 'height': 700})
        chartLog.set_plotarea({'layout': {'x': 0.15, 'y': 0.15, 'width': 0.75, 'height': 0.70}})
        chartLog.set_legend({'font': {'size': 12}})
        
        # Insert chart into worksheet
        plotLogWorksheet.insert_chart('A1', chartLog)

        # Create PSD plot worksheet
        psd_plot_sheet_name = 'PSD_Plot'
        counter = 1
        while psd_plot_sheet_name in used_worksheet_names:
            psd_plot_sheet_name = f'PSD_Plot_{counter}'
            counter += 1
        used_worksheet_names.add(psd_plot_sheet_name)
        psdPlotWorksheet = summaryWorkbook.add_worksheet(psd_plot_sheet_name)
        
        # Create scatter chart with straight lines for PSD
        chartPSD = summaryWorkbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
        
        # First add all non-NoLeak series (colorful lines in background)
        color_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' not in series_info['sheet_name']:
                series_config = {
                    'name': series_info['sheet_name'],
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 9, series_info['data_points'], 9],      # PSD_AVG column (column J)
                    'line': {'width': 2}
                }
                # Apply distinguishable color for regular series
                regular_color = distinguishable_colors[color_index % len(distinguishable_colors)]
                series_config['line']['color'] = regular_color
                color_index += 1
                chartPSD.add_series(series_config)
        
        # Then add all NoLeak series (grey lines in front)
        grey_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' in series_info['sheet_name']:
                series_config = {
                    'name': series_info['sheet_name'],
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 9, series_info['data_points'], 9],      # PSD_AVG column (column J)
                    'line': {'width': 2}
                }
                # Apply grey color in ascending order (darker to lighter)
                grey_color = grey_colors[grey_index % len(grey_colors)]
                series_config['line']['color'] = grey_color
                grey_index += 1
                chartPSD.add_series(series_config)
        
        # Configure PSD chart with logarithmic frequency axis
        chartPSD.set_title({'name': 'PSD Frequency vs Average Values', 'name_font': {'size': 12}})
        chartPSD.set_x_axis({
            'name': 'Frequency (Hz)',
            'log_base': 10,
            'label_position': 'low',
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartPSD.set_y_axis({
            'name': 'PSD Average Values',
            'label_position': 'low',
            'name_layout': {'x': 0.02, 'y': 0.5},
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartPSD.set_size({'width': 1400, 'height': 700})
        chartPSD.set_plotarea({'layout': {'x': 0.15, 'y': 0.15, 'width': 0.75, 'height': 0.70}})
        chartPSD.set_legend({'font': {'size': 12}})
        
        # Insert PSD chart into worksheet
        psdPlotWorksheet.insert_chart('A1', chartPSD)

        # Create SNR plot worksheet
        snr_plot_sheet_name = 'SNR_Plot'
        counter = 1
        while snr_plot_sheet_name in used_worksheet_names:
            snr_plot_sheet_name = f'SNR_Plot_{counter}'
            counter += 1
        used_worksheet_names.add(snr_plot_sheet_name)
        snrPlotWorksheet = summaryWorkbook.add_worksheet(snr_plot_sheet_name)
        
        # Create scatter chart with straight lines for SNR
        chartSNR = summaryWorkbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
        
        # First add all non-NoLeak series (colorful lines in background)
        color_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' not in series_info['sheet_name']:
                series_config = {
                    'name': series_info['sheet_name'],
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 10, series_info['data_points'], 10],     # SNR column (column K)
                    'line': {'width': 2}
                }
                # Apply distinguishable color for regular series
                regular_color = distinguishable_colors[color_index % len(distinguishable_colors)]
                series_config['line']['color'] = regular_color
                color_index += 1
                chartSNR.add_series(series_config)
        
        # Then add all NoLeak series as reference (flat line at 0 dB)
        grey_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' in series_info['sheet_name']:
                # Create SNR reference line for NoLeak measurements
                series_config = {
                    'name': series_info['sheet_name'] + ' (Reference)',
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 10, series_info['data_points'], 10],     # SNR column (column K)
                    'line': {'width': 1}
                }
                # Apply grey color
                grey_color = grey_colors[grey_index % len(grey_colors)]
                series_config['line']['color'] = grey_color
                grey_index += 1
                chartSNR.add_series(series_config)
        
        # Configure SNR chart
        chartSNR.set_title({'name': 'Signal-to-Noise Ratio (PSD) vs Frequency', 'name_font': {'size': 12}})
        chartSNR.set_x_axis({
            'name': 'Frequency (Hz)',
            'log_base': 10,
            'label_position': 'low',
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartSNR.set_y_axis({
            'name': 'SNR (Linear)',
            'label_position': 'low',
            'name_layout': {'x': 0.02, 'y': 0.5},
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartSNR.set_size({'width': 1400, 'height': 700})
        chartSNR.set_plotarea({'layout': {'x': 0.15, 'y': 0.15, 'width': 0.75, 'height': 0.70}})
        chartSNR.set_legend({'font': {'size': 12}})
        
        # Insert SNR chart into worksheet
        snrPlotWorksheet.insert_chart('A1', chartSNR)

        # Create File-Specific SNR plot worksheet
        snr_distance_sheet_name = 'SNR_File_Specific'
        counter = 1
        while snr_distance_sheet_name in used_worksheet_names:
            snr_distance_sheet_name = f'SNR_File_Specific_{counter}'
            counter += 1
        used_worksheet_names.add(snr_distance_sheet_name)
        snrDistanceWorksheet = summaryWorkbook.add_worksheet(snr_distance_sheet_name)
        
        # Calculate file-specific SNR data
        if analysis_data:
            # Group data by WAV file base name (without _leak/_noleak suffix)
            wav_file_groups = {}
            
            for data in analysis_data:
                # Extract base filename without _leak/_noleak suffix
                base_filename = re.sub(r'_(leak|noleak)$', '', data['filename'])
                
                if base_filename not in wav_file_groups:
                    wav_file_groups[base_filename] = {'noleak': [], 'all_measurements': []}
                
                wav_file_groups[base_filename]['all_measurements'].append(data)
                if data['is_noleak']:
                    wav_file_groups[base_filename]['noleak'].append(data)
            
            # Calculate file-specific SNR for each WAV file group
            for base_filename, group in wav_file_groups.items():
                if group['noleak']:
                    # Create improved WAV file-specific noise floor using robust methods
                    wav_file_noise_floor = improved_noise_floor_estimation(
                        [data['psd_avg'] for data in group['noleak']], 
                        method='median',  # Use median for file-specific estimation (more robust for smaller samples)
                        min_noise_threshold=1e-12
                    )
                    
                    # Create improved WAV file-specific FFT-based noise floor
                    wav_file_fft_noise_floor = improved_noise_floor_estimation(
                        [data['fft_abs_min'] for data in group['noleak']], 
                        method='median',
                        min_noise_threshold=1e-12
                    )
                    
                    # Calculate improved SNR for all measurements from this WAV file
                    for data in group['all_measurements']:
                        # Calculate robust file-specific SNR with clipping
                        snr_file = calculate_robust_snr(
                            data['psd_avg'], 
                            wav_file_noise_floor,
                            method='clipped_linear',
                            max_snr_db=60,
                            min_snr_db=-40
                        )
                        
                        # Calculate robust file-specific FFT-based SNR with clipping
                        snr_fft_file = calculate_robust_snr(
                            data['fft_abs_min'], 
                            wav_file_fft_noise_floor,
                            method='clipped_linear',
                            max_snr_db=60,
                            min_snr_db=-40
                        )
                        
                        # Update the worksheet with file-specific SNR values
                        sheet_name = data['filename']
                        
                        # Find the corresponding sanitized sheet name
                        sanitized_sheet_name = None
                        for series_info in chart_series_info:
                            if series_info['filename'] == sheet_name:
                                sanitized_sheet_name = series_info['sheet_name']
                                break
                        
                        if sanitized_sheet_name:
                            # Find and update the corresponding worksheet
                            for ws in summaryWorkbook.worksheets():
                                if ws.get_name() == sanitized_sheet_name:
                                    # Update file-specific SNR column (column L, index 11)
                                    for j in range(len(snr_file)):
                                        ws.write(j+1, 11, sanitize_excel_value(snr_file[j]))
                                    # Update file-specific FFT-based SNR column (column N, index 13)
                                    for j in range(len(snr_fft_file)):
                                        ws.write(j+1, 13, sanitize_excel_value(snr_fft_file[j]))
                                    break
        
        # Create scatter chart with straight lines for file-specific SNR
        chartSNRDist = summaryWorkbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
        
        # First add all non-NoLeak series (colorful lines in background)
        color_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' not in series_info['sheet_name']:
                series_config = {
                    'name': series_info['sheet_name'],
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 11, series_info['data_points'], 11],     # SNR_File column (column L)
                    'line': {'width': 2}
                }
                # Apply distinguishable color for regular series
                regular_color = distinguishable_colors[color_index % len(distinguishable_colors)]
                series_config['line']['color'] = regular_color
                color_index += 1
                chartSNRDist.add_series(series_config)
        
        # Then add all NoLeak series as reference
        grey_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' in series_info['sheet_name']:
                # Create SNR reference line for NoLeak measurements (should be around 1.0 for file-specific linear)
                series_config = {
                    'name': series_info['sheet_name'] + ' (File Ref)',
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 11, series_info['data_points'], 11],     # SNR_File column (column L)
                    'line': {'width': 1}
                }
                # Apply grey color
                grey_color = grey_colors[grey_index % len(grey_colors)]
                series_config['line']['color'] = grey_color
                grey_index += 1
                chartSNRDist.add_series(series_config)
        
        # Configure file-specific SNR chart
        chartSNRDist.set_title({'name': 'File-Specific Signal-to-Noise Ratio (PSD) vs Frequency', 'name_font': {'size': 12}})
        chartSNRDist.set_x_axis({
            'name': 'Frequency (Hz)',
            'log_base': 10,
            'label_position': 'low',
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartSNRDist.set_y_axis({
            'name': 'SNR File-Specific (Linear)',
            'label_position': 'low',
            'name_layout': {'x': 0.02, 'y': 0.5},
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartSNRDist.set_size({'width': 1400, 'height': 700})
        chartSNRDist.set_plotarea({'layout': {'x': 0.15, 'y': 0.15, 'width': 0.75, 'height': 0.70}})
        chartSNRDist.set_legend({'font': {'size': 12}})
        
        # Insert file-specific SNR chart into worksheet
        snrDistanceWorksheet.insert_chart('A1', chartSNRDist)
        
        # Create FFT-based SNR plot worksheet
        snr_fft_sheet_name = 'SNR_FFT'
        counter = 1
        while snr_fft_sheet_name in used_worksheet_names:
            snr_fft_sheet_name = f'SNR_FFT_{counter}'
            counter += 1
        used_worksheet_names.add(snr_fft_sheet_name)
        snrFFTWorksheet = summaryWorkbook.add_worksheet(snr_fft_sheet_name)
        
        # Create scatter chart with straight lines for FFT-based SNR
        chartSNRFFT = summaryWorkbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
        
        # First add all non-NoLeak series (colorful lines in background)
        color_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' not in series_info['sheet_name']:
                series_config = {
                    'name': series_info['sheet_name'],
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 12, series_info['data_points'], 12],     # SNR_FFT column (column M)
                    'line': {'width': 2}
                }
                # Apply distinguishable color for regular series
                regular_color = distinguishable_colors[color_index % len(distinguishable_colors)]
                series_config['line']['color'] = regular_color
                color_index += 1
                chartSNRFFT.add_series(series_config)
        
        # Then add all NoLeak series (grey lines in front)
        grey_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' in series_info['sheet_name']:
                # Create SNR reference line for NoLeak measurements
                series_config = {
                    'name': series_info['sheet_name'] + ' (Reference)',
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 12, series_info['data_points'], 12],     # SNR_FFT column (column M)
                    'line': {'width': 1}
                }
                # Apply grey color
                grey_color = grey_colors[grey_index % len(grey_colors)]
                series_config['line']['color'] = grey_color
                grey_index += 1
                chartSNRFFT.add_series(series_config)
        
        # Configure FFT-based SNR chart
        chartSNRFFT.set_title({'name': 'Signal-to-Noise Ratio (FFT) vs Frequency', 'name_font': {'size': 12}})
        chartSNRFFT.set_x_axis({
            'name': 'Frequency (Hz)',
            'log_base': 10,
            'label_position': 'low',
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartSNRFFT.set_y_axis({
            'name': 'SNR FFT (Linear)',
            'label_position': 'low',
            'name_layout': {'x': 0.02, 'y': 0.5},
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartSNRFFT.set_size({'width': 1400, 'height': 700})
        chartSNRFFT.set_plotarea({'layout': {'x': 0.15, 'y': 0.15, 'width': 0.75, 'height': 0.70}})
        chartSNRFFT.set_legend({'font': {'size': 12}})
        
        # Insert FFT-based SNR chart into worksheet
        snrFFTWorksheet.insert_chart('A1', chartSNRFFT)
        
        # Create File-Specific FFT-based SNR plot worksheet
        snr_fft_file_sheet_name = 'SNR_FFT_File_Specific'
        counter = 1
        while snr_fft_file_sheet_name in used_worksheet_names:
            snr_fft_file_sheet_name = f'SNR_FFT_File_Specific_{counter}'
            counter += 1
        used_worksheet_names.add(snr_fft_file_sheet_name)
        snrFFTFileWorksheet = summaryWorkbook.add_worksheet(snr_fft_file_sheet_name)
        
        # Create scatter chart with straight lines for file-specific FFT-based SNR
        chartSNRFFTFile = summaryWorkbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
        
        # First add all non-NoLeak series (colorful lines in background)
        color_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' not in series_info['sheet_name']:
                series_config = {
                    'name': series_info['sheet_name'],
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 13, series_info['data_points'], 13],     # SNR_FFT_File column (column N)
                    'line': {'width': 2}
                }
                # Apply distinguishable color for regular series
                regular_color = distinguishable_colors[color_index % len(distinguishable_colors)]
                series_config['line']['color'] = regular_color
                color_index += 1
                chartSNRFFTFile.add_series(series_config)
        
        # Then add all NoLeak series as reference
        grey_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' in series_info['sheet_name']:
                # Create SNR reference line for NoLeak measurements (should be around 1.0 for file-specific linear)
                series_config = {
                    'name': series_info['sheet_name'] + ' (File Ref)',
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 13, series_info['data_points'], 13],     # SNR_FFT_File column (column N)
                    'line': {'width': 1}
                }
                # Apply grey color
                grey_color = grey_colors[grey_index % len(grey_colors)]
                series_config['line']['color'] = grey_color
                grey_index += 1
                chartSNRFFTFile.add_series(series_config)
        
        # Configure file-specific FFT-based SNR chart
        chartSNRFFTFile.set_title({'name': 'File-Specific Signal-to-Noise Ratio (FFT) vs Frequency', 'name_font': {'size': 12}})
        chartSNRFFTFile.set_x_axis({
            'name': 'Frequency (Hz)',
            'log_base': 10,
            'label_position': 'low',
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartSNRFFTFile.set_y_axis({
            'name': 'SNR FFT File-Specific (Linear)',
            'label_position': 'low',
            'name_layout': {'x': 0.02, 'y': 0.5},
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartSNRFFTFile.set_size({'width': 1400, 'height': 700})
        chartSNRFFTFile.set_plotarea({'layout': {'x': 0.15, 'y': 0.15, 'width': 0.75, 'height': 0.70}})
        chartSNRFFTFile.set_legend({'font': {'size': 12}})
        
        # Insert file-specific FFT-based SNR chart into worksheet
        snrFFTFileWorksheet.insert_chart('A1', chartSNRFFTFile)
    
        # Create Leak Detection Results worksheet
        if leak_detection_results and isinstance(leak_detection_results, dict):
            leak_detection_sheet_name = 'Leak_Detection'
            counter = 1
            while leak_detection_sheet_name in used_worksheet_names:
                leak_detection_sheet_name = f'Leak_Detection_{counter}'
                counter += 1
            used_worksheet_names.add(leak_detection_sheet_name)
            leakDetectionWorksheet = summaryWorkbook.add_worksheet(leak_detection_sheet_name)
            
            # Write headers
            headers = ['Measurement', 'Leak Probability', 'Composite Score', 'Statistical Score', 'Frequency Band Score', 
                      'Power Ratio Score', 'Max Power Increase (dB)', 'Detection Ratio (%)', 'Max Exceedance Ratio']
            
            for col, header in enumerate(headers):
                leakDetectionWorksheet.write(0, col, header)
            
            # Write detection results
            row = 1
            for filename, result in leak_detection_results.items():
                leakDetectionWorksheet.write(row, 0, filename)
                leakDetectionWorksheet.write(row, 1, result['leak_probability'])
                leakDetectionWorksheet.write(row, 2, round(result['composite_score'], 1))
                leakDetectionWorksheet.write(row, 3, round(result['individual_scores']['statistical'], 1))
                leakDetectionWorksheet.write(row, 4, round(result['individual_scores']['frequency_bands'], 1))
                leakDetectionWorksheet.write(row, 5, round(result['individual_scores']['power_ratio'], 1))
                leakDetectionWorksheet.write(row, 6, round(result['detailed_results']['power_ratio']['max_power_increase_dB'], 1))
                leakDetectionWorksheet.write(row, 7, round(result['detailed_results']['statistical']['detection_ratio'] * 100, 1))
                leakDetectionWorksheet.write(row, 8, round(result['detailed_results']['statistical']['max_exceedance_ratio'], 2))
                row += 1
            
            # Add frequency band analysis details
            row += 2
            leakDetectionWorksheet.write(row, 0, 'Frequency Band Analysis Details:')
            row += 1
            
            # Headers for frequency band details
            band_headers = ['Measurement', 'Frequency Band', 'SNR Ratio', 'Z-Score', 'Leak Detected']
            for col, header in enumerate(band_headers):
                leakDetectionWorksheet.write(row, col, header)
            row += 1
            
            # Write frequency band details
            for filename, result in leak_detection_results.items():
                for band_name, band_result in result['detailed_results']['frequency_bands'].items():
                    leakDetectionWorksheet.write(row, 0, filename)
                    leakDetectionWorksheet.write(row, 1, band_name.replace('_', ' ').replace('band ', ''))
                    leakDetectionWorksheet.write(row, 2, round(band_result['snr_ratio'], 2))
                    leakDetectionWorksheet.write(row, 3, round(band_result['z_score'], 2))
                    leakDetectionWorksheet.write(row, 4, 'YES' if band_result['leak_detected'] else 'NO')
                    row += 1
            
            # Add summary statistics
            row += 2
            leakDetectionWorksheet.write(row, 0, 'Detection Summary:')
            row += 1
            
            # Count detections by probability
            high_prob = sum(1 for r in leak_detection_results.values() if r['leak_probability'] == 'HIGH')
            medium_prob = sum(1 for r in leak_detection_results.values() if r['leak_probability'] == 'MEDIUM')
            low_prob = sum(1 for r in leak_detection_results.values() if r['leak_probability'] == 'LOW')
            
            leakDetectionWorksheet.write(row, 0, 'High Probability Leaks:')
            leakDetectionWorksheet.write(row, 1, high_prob)
            row += 1
            leakDetectionWorksheet.write(row, 0, 'Medium Probability Leaks:')
            leakDetectionWorksheet.write(row, 1, medium_prob)
            row += 1
            leakDetectionWorksheet.write(row, 0, 'Low Probability Leaks:')
            leakDetectionWorksheet.write(row, 1, low_prob)
            row += 1
            
            # Average scores
            if leak_detection_results:
                avg_composite = np.mean([r['composite_score'] for r in leak_detection_results.values()])
                leakDetectionWorksheet.write(row, 0, 'Average Composite Score:')
                leakDetectionWorksheet.write(row, 1, round(avg_composite, 1))

        # Create File-Specific Leak Detection Results worksheet
        if leak_detection_distance_specific and isinstance(leak_detection_distance_specific, dict):
            distance_detection_sheet_name = 'File_Specific_Detection'
            counter = 1
            while distance_detection_sheet_name in used_worksheet_names:
                distance_detection_sheet_name = f'File_Specific_Detection_{counter}'
                counter += 1
            used_worksheet_names.add(distance_detection_sheet_name)
            distanceDetectionWorksheet = summaryWorkbook.add_worksheet(distance_detection_sheet_name)
            
            # Write headers
            headers = ['Measurement', 'Distance (m)', 'Baseline Available', 'Baseline Count', 'Baseline Files', 
                      'Leak Probability', 'Composite Score', 'Statistical Score', 'Frequency Band Score', 
                      'Power Ratio Score', 'Max Power Increase (dB)', 'Detection Ratio (%)', 'Max Exceedance Ratio']
            
            for col, header in enumerate(headers):
                distanceDetectionWorksheet.write(0, col, header)
            
            # Write detection results
            row = 1
            for filename, result in leak_detection_distance_specific.items():
                distanceDetectionWorksheet.write(row, 0, filename)
                distanceDetectionWorksheet.write(row, 1, result['distance'])
                
                if not result.get('baseline_available', True):
                    # Handle case where no baseline is available
                    distanceDetectionWorksheet.write(row, 2, 'NO')
                    distanceDetectionWorksheet.write(row, 3, 0)
                    distanceDetectionWorksheet.write(row, 4, 'N/A')
                    distanceDetectionWorksheet.write(row, 5, result.get('error', 'No baseline'))
                    for col in range(6, len(headers)):
                        distanceDetectionWorksheet.write(row, col, 'N/A')
                else:
                    # Normal case with baseline available
                    distanceDetectionWorksheet.write(row, 2, 'YES')
                    distanceDetectionWorksheet.write(row, 3, result['baseline_measurements'])
                    distanceDetectionWorksheet.write(row, 4, ', '.join(result['baseline_files']))
                    distanceDetectionWorksheet.write(row, 5, result['leak_probability'])
                    distanceDetectionWorksheet.write(row, 6, round(result['composite_score'], 1))
                    distanceDetectionWorksheet.write(row, 7, round(result['individual_scores']['statistical'], 1))
                    distanceDetectionWorksheet.write(row, 8, round(result['individual_scores']['frequency_bands'], 1))
                    distanceDetectionWorksheet.write(row, 9, round(result['individual_scores']['power_ratio'], 1))
                    distanceDetectionWorksheet.write(row, 10, round(result['detailed_results']['power_ratio']['max_power_increase_dB'], 1))
                    distanceDetectionWorksheet.write(row, 11, round(result['detailed_results']['statistical']['detection_ratio'] * 100, 1))
                    distanceDetectionWorksheet.write(row, 12, round(result['detailed_results']['statistical']['max_exceedance_ratio'], 2))
                
                row += 1
            
            # Add file-specific frequency band analysis details
            row += 2
            distanceDetectionWorksheet.write(row, 0, 'File-Specific Frequency Band Analysis:')
            row += 1
            
            # Headers for frequency band details
            band_headers = ['Measurement', 'Distance (m)', 'Frequency Band', 'SNR Ratio', 'Z-Score', 'Leak Detected']
            for col, header in enumerate(band_headers):
                distanceDetectionWorksheet.write(row, col, header)
            row += 1
            
            # Write frequency band details
            for filename, result in leak_detection_distance_specific.items():
                if result.get('baseline_available', True) and 'detailed_results' in result:
                    for band_name, band_result in result['detailed_results']['frequency_bands'].items():
                        distanceDetectionWorksheet.write(row, 0, filename)
                        distanceDetectionWorksheet.write(row, 1, result['distance'])
                        distanceDetectionWorksheet.write(row, 2, band_name.replace('_', ' ').replace('band ', ''))
                        distanceDetectionWorksheet.write(row, 3, round(band_result['snr_ratio'], 2))
                        distanceDetectionWorksheet.write(row, 4, round(band_result['z_score'], 2))
                        distanceDetectionWorksheet.write(row, 5, 'YES' if band_result['leak_detected'] else 'NO')
                        row += 1
            
            # Add file-specific summary statistics
            row += 2
            distanceDetectionWorksheet.write(row, 0, 'File-Specific Detection Summary:')
            row += 1
            
            # Group results by distance for summary
            distance_summary = {}
            for filename, result in leak_detection_distance_specific.items():
                distance = result['distance']
                if distance not in distance_summary:
                    distance_summary[distance] = {'high': 0, 'medium': 0, 'low': 0, 'no_baseline': 0, 'total': 0}
                
                distance_summary[distance]['total'] += 1
                
                if not result.get('baseline_available', True):
                    distance_summary[distance]['no_baseline'] += 1
                else:
                    prob = result['leak_probability']
                    if prob == 'HIGH':
                        distance_summary[distance]['high'] += 1
                    elif prob == 'MEDIUM':
                        distance_summary[distance]['medium'] += 1
                    else:
                        distance_summary[distance]['low'] += 1
            
            # Write summary by distance
            summary_headers = ['Distance (m)', 'Total Measurements', 'High Prob', 'Medium Prob', 'Low Prob', 'No Baseline']
            for col, header in enumerate(summary_headers):
                distanceDetectionWorksheet.write(row, col, header)
            row += 1
            
            for distance in sorted(distance_summary.keys()):
                summary = distance_summary[distance]
                distanceDetectionWorksheet.write(row, 0, distance)
                distanceDetectionWorksheet.write(row, 1, summary['total'])
                distanceDetectionWorksheet.write(row, 2, summary['high'])
                distanceDetectionWorksheet.write(row, 3, summary['medium'])
                distanceDetectionWorksheet.write(row, 4, summary['low'])
                distanceDetectionWorksheet.write(row, 5, summary['no_baseline'])
                row += 1
    
    # Close workbook for this folder
    summaryWorkbook.close()
    print(f"✓ Analysis complete for {subfolder_name}: {summary_filename}")

# Sort by numeric value before "m" (if present)
def extract_number_before_m(filename):
    match = re.search(r'(\d+)m', filename)
    return int(match.group(1)) if match else float('inf')

# Store all folders' analysis data for summary comparison
all_folders_data = {}

# Main processing loop - process each folder separately
for subfolder in subfolders:
    subfolder_path = os.path.join(path, subfolder)
    all_wav_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.wav')]
    
    # Count only WAV files with distance pattern
    wav_files_with_distance = [f for f in all_wav_files if re.search(r'\d+m', f)]
    total_wav_files += len(wav_files_with_distance)
    
    print(f"\nProcessing folder: {subfolder} ({len(wav_files_with_distance)} WAV files with distance pattern)")
    
    if len(wav_files_with_distance) == 0:
        print(f"  No WAV files with distance pattern found in {subfolder} - skipping folder")
        continue
    
    # Process WAV files in this subfolder
    folder_data = process_wav_files_in_folder(subfolder_path)
    total_segments += len(folder_data)
    
    # Store folder data for summary comparison
    all_folders_data[subfolder] = folder_data
    
    # Process and analyze this folder's data
    process_folder_analysis(subfolder_path, subfolder, folder_data)

# Create summary comparison file
def create_summary_comparison(all_folders_data, base_path):
    """Create a summary comparison Excel file across all subfolders"""
    
    if not all_folders_data:
        print("No data available for summary comparison")
        return
    
    # Create summary comparison file
    summary_path = os.path.join(base_path, "summary_comparison.xlsx")
    summaryComparisonWorkbook = xlsxwriter.Workbook(summary_path, {'nan_inf_to_errors': True})
    
    print(f"\nCreating summary comparison file: {summary_path}")
    
    # Collect all analysis data from all folders
    all_analysis_data = []
    folder_mapping = {}  # Maps filename to folder name
    
    for folder_name, folder_data in all_folders_data.items():
        for fName, DataFrame in folder_data:
            # Process the data similar to individual folder processing
            DataFrameSize = len(DataFrame)
            
            if n_AVG <= DataFrameSize//n_samples:
                time_Array = DataFrame[0][0:n_samples].to_numpy()
                
                N = len(time_Array)
                
                # Calculate sampling rate
                listSampleRate = []
                for i in range(1, N):
                    sampleRate = time_Array[i] - time_Array[i-1]
                    listSampleRate.append(sampleRate)
                
                st = sum(listSampleRate) / len(listSampleRate)
                st = round(st, 9)
                fs = 1 / st
                
                fftFreq = np.fft.fftfreq(N, st)
                hanning_window = np.hanning(N)
                
                # Process FFT data
                fft = []
                for j in range(n_AVG):
                    data_Array = DataFrame[1][j*n_samples:(j+1)*n_samples].to_numpy()
                    data_Array = np.transpose(data_Array)
                    windowed_data = data_Array * hanning_window
                    fft.append(np.fft.fft(windowed_data))
                
                fftAbs = np.abs(fft)/N*2*2
                fftAbs_MIN = np.min(fftAbs, axis=0)
                
                # Calculate PSD
                hanning_correction = 8/3
                psd = []
                for j in range(n_AVG):
                    psd_single = (np.abs(fft[j])**2) / (N * fs) * hanning_correction
                    psd.append(psd_single)
                
                psd_AVG = np.mean(psd, axis=0)
                
                # Store analysis data
                analysis_entry = {
                    'folder': folder_name,
                    'filename': fName,
                    'frequency': fftFreq[:N//2],
                    'psd_avg': psd_AVG[:N//2],
                    'fft_abs_min': fftAbs_MIN[:N//2],
                    'is_noleak': 'noleak' in fName.lower()
                }
                
                all_analysis_data.append(analysis_entry)
                folder_mapping[fName] = folder_name
    
    # Create FFT Bands SNR Comparison Table
    create_fft_bands_snr_comparison(summaryComparisonWorkbook, all_analysis_data)
    
    # Create overall summary statistics
    create_overall_summary_statistics(summaryComparisonWorkbook, all_analysis_data)
    
    # Create SNR vs Frequency Bands plot
    create_snr_frequency_bands_plot(summaryComparisonWorkbook, all_analysis_data)
    
    # Create file-specific SNR analysis tabs
    create_file_specific_fft_bands_snr_comparison(summaryComparisonWorkbook, all_analysis_data)
    create_file_specific_overall_summary_statistics(summaryComparisonWorkbook, all_analysis_data)
    create_file_specific_snr_frequency_bands_plot(summaryComparisonWorkbook, all_analysis_data)
    
    # Close the workbook
    summaryComparisonWorkbook.close()
    print(f"Summary comparison file created successfully: {summary_path}")

def create_fft_bands_snr_comparison(workbook, all_analysis_data):
    """Create FFT bands SNR comparison table across all folders"""
    
    # Create worksheet
    worksheet = workbook.add_worksheet('FFT_Bands_SNR_Comparison')
    
    # Define frequency bands for analysis (same as leak detection)
    frequency_bands = [
        # Very low frequency bands (1-100 Hz) - 10 bands
        ('Ultra-Low (1-10Hz)', 1, 10),       # Ultra-low frequency structural
        ('Very Low (10-20Hz)', 10, 20),      # Very low frequency vibrations
        ('Low Structural (20-30Hz)', 20, 30),      # Low frequency structural response
        ('Mechanical (30-40Hz)', 30, 40),      # Mechanical vibrations
        ('Power Harmonics (40-50Hz)', 40, 50),      # Power line and mechanical harmonics
        ('Power Frequency (50-60Hz)', 50, 60),      # Power frequency range
        ('Post-Power (60-70Hz)', 60, 70),      # Post-power frequency
        ('Low Acoustic (70-80Hz)', 70, 80),      # Low acoustic range
        ('Pre-Acoustic (80-90Hz)', 80, 90),      # Pre-acoustic range
        ('Low Acoustic Transition (90-100Hz)', 90, 100),     # Low acoustic transition
        # Refined frequency bands
        # 100-500 Hz range in 50 Hz increments
        ('Low Structural 1 (100-150Hz)', 100, 150),    # Low structural range 1
        ('Low Structural 2 (150-200Hz)', 150, 200),    # Low structural range 2
        ('Low Structural 3 (200-250Hz)', 200, 250),    # Low structural range 3
        ('Low Structural 4 (250-300Hz)', 250, 300),    # Low structural range 4
        ('Low Structural 5 (300-350Hz)', 300, 350),    # Low structural range 5
        ('Low Structural 6 (350-400Hz)', 350, 400),    # Low structural range 6
        ('Low Structural 7 (400-450Hz)', 400, 450),    # Low structural range 7
        ('Low Structural 8 (450-500Hz)', 450, 500),    # Low structural range 8
        # 500-2000 Hz range in 1000 Hz jumps
        ('Mid Frequency 1 (500-1500Hz)', 500, 1500),   # Mid frequency acoustic emissions 1
        ('Mid Frequency 2 (1500-2000Hz)', 1500, 2000),  # Mid frequency acoustic emissions 2
        # Original higher frequency bands
        ('High Frequency (2000-8000Hz)', 2000, 8000),  # High frequency turbulence
        ('Ultrasonic (8000-20000Hz)', 8000, 20000)  # Ultrasonic range
    ]
    
    # Group data by folder
    folder_groups = {}
    for data in all_analysis_data:
        folder = data['folder']
        if folder not in folder_groups:
            folder_groups[folder] = {'noleak': [], 'leak': []}
        
        if data['is_noleak']:
            folder_groups[folder]['noleak'].append(data)
        else:
            folder_groups[folder]['leak'].append(data)
    
    # Get sorted folder names for column headers - sort by numbers followed by "lh" (case-insensitive)
    def extract_lh_number(folder_name):
        # Try to extract number followed by "lh" or "LH" (case-insensitive)
        lh_match = re.search(r'(\d+)lh', folder_name, re.IGNORECASE)
        if lh_match:
            return int(lh_match.group(1))
        else:
            return float('inf')  # Put folders without lh pattern at the end
    
    sorted_folders = sorted(folder_groups.keys(), key=lambda x: (extract_lh_number(x), x))
    
    # Create a structure to hold all measurement/frequency band combinations (collect from all folders)
    measurement_band_combinations = []
    unique_measurements = set()
    
    # Collect all unique measurements from all folders using worksheet tab names
    for folder_name in sorted_folders:
        folder_data = folder_groups[folder_name]
        if folder_data['noleak'] and folder_data['leak']:
            leak_measurements = folder_data['leak']
            for measurement in leak_measurements:
                # Extract distance from filename
                distance_match = re.search(r'(\d+)m', measurement['filename'])
                distance = int(distance_match.group(1)) if distance_match else 0
                
                # Create worksheet tab name (same logic as in individual folder processing)
                filename = measurement['filename']
                
                # Extract distance and any additional word after it
                distance_match = re.search(r'(\d+)m(?:[_\s]?([A-Za-z]+))?', filename)
                if distance_match:
                    distance_str = f"{distance_match.group(1)}m"
                    additional_word = distance_match.group(2)  # The word after 'm' (if any)
                    
                    # Build distance part with additional word if present
                    if additional_word and additional_word.lower() not in ['leak', 'noleak']:
                        distance_part = f"{distance_str}_{additional_word}"
                    else:
                        distance_part = distance_str
                else:
                    distance_part = "Unknown"
                
                # Determine leak/noleak designation
                if '_leak' in filename.lower():
                    leak_designation = "Leak"
                elif '_noleak' in filename.lower():
                    leak_designation = "NoLeak"
                else:
                    leak_designation = "Unknown"
                
                # Create worksheet name: "5m_sensor_Leak", "10m_test_NoLeak", etc.
                worksheet_name = f"{distance_part}_{leak_designation}"
                
                # Add to unique measurements set using worksheet name
                measurement_key = (worksheet_name, distance)
                unique_measurements.add(measurement_key)
    
    # Create combinations for all unique measurements and all frequency bands
    # Sort measurements by distance first, then by worksheet name (same as tab order)
    sorted_measurements = sorted(unique_measurements, key=lambda x: (x[1], x[0]))  # Sort by distance, then worksheet name
    
    for worksheet_name, distance in sorted_measurements:
        for band_name, freq_min, freq_max in frequency_bands:
            combination = (worksheet_name, distance, band_name)
            measurement_band_combinations.append(combination)
    
    # Write headers - basic info columns followed by folder SNR columns
    headers = ['Measurement', 'Distance (m)', 'Frequency Band']
    for folder_name in sorted_folders:
        headers.append(f'{folder_name} SNR')
    
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)
    
    # Calculate SNR values for each folder
    folder_snr_data = {}
    for folder_name in sorted_folders:
        folder_data = folder_groups[folder_name]
        if not folder_data['noleak'] or not folder_data['leak']:
            continue
        
        # Calculate folder-specific FFT noise floor
        folder_fft_noise = []
        for noleak_data in folder_data['noleak']:
            folder_fft_noise.append(noleak_data['fft_abs_min'])
        
        if folder_fft_noise:
            # Use improved noise floor estimation for summary comparison
            folder_noise_floor = improved_noise_floor_estimation(
                folder_fft_noise,
                method='percentile',
                percentile=10,
                min_noise_threshold=1e-12
            )
            
            folder_snr_data[folder_name] = {}
            
            # Process leak measurements
            leak_measurements = folder_data['leak']
            for measurement in leak_measurements:
                # Calculate robust SNR for each frequency band
                freq = measurement['frequency']
                signal = measurement['fft_abs_min']
                snr_values = calculate_robust_snr(
                    signal, 
                    folder_noise_floor,
                    method='clipped_linear',
                    max_snr_db=60,
                    min_snr_db=-40
                )
                
                # Create worksheet tab name (same logic as in individual folder processing)
                filename = measurement['filename']
                
                # Extract distance and any additional word after it
                distance_match = re.search(r'(\d+)m(?:[_\s]?([A-Za-z]+))?', filename)
                if distance_match:
                    distance_str = f"{distance_match.group(1)}m"
                    additional_word = distance_match.group(2)  # The word after 'm' (if any)
                    
                    # Build distance part with additional word if present
                    if additional_word and additional_word.lower() not in ['leak', 'noleak']:
                        distance_part = f"{distance_str}_{additional_word}"
                    else:
                        distance_part = distance_str
                else:
                    distance_part = "Unknown"
                
                # Determine leak/noleak designation
                if '_leak' in filename.lower():
                    leak_designation = "Leak"
                elif '_noleak' in filename.lower():
                    leak_designation = "NoLeak"
                else:
                    leak_designation = "Unknown"
                
                # Create worksheet name: "5m_sensor_Leak", "10m_test_NoLeak", etc.
                worksheet_name = f"{distance_part}_{leak_designation}"
                
                for band_name, freq_min, freq_max in frequency_bands:
                    # Find frequency indices for this band
                    band_mask = (freq >= freq_min) & (freq <= freq_max)
                    
                    if np.any(band_mask):
                        band_snr = np.mean(snr_values[band_mask])
                        key = (worksheet_name, band_name)
                        folder_snr_data[folder_name][key] = band_snr
    
    # Write data rows
    row = 1
    for worksheet_name, distance, band_name in measurement_band_combinations:
        # Write basic info
        worksheet.write(row, 0, worksheet_name)
        worksheet.write(row, 1, distance)
        worksheet.write(row, 2, band_name)
        
        # Write SNR values for each folder
        for col_idx, folder_name in enumerate(sorted_folders):
            if folder_name in folder_snr_data:
                key = (worksheet_name, band_name)
                if key in folder_snr_data[folder_name]:
                    snr_value = folder_snr_data[folder_name][key]
                    worksheet.write(row, 3 + col_idx, round(snr_value, 3))
                else:
                    worksheet.write(row, 3 + col_idx, 'N/A')
            else:
                worksheet.write(row, 3 + col_idx, 'No Data')
        
        row += 1
    
    # Add summary statistics section
    row += 2
    worksheet.write(row, 0, 'SUMMARY STATISTICS')
    row += 2
    
    # Calculate folder averages for each frequency band
    worksheet.write(row, 0, 'Average SNR by Frequency Band:')
    row += 1
    
    # Write summary headers
    summary_headers = ['Frequency Band']
    for folder_name in sorted_folders:
        summary_headers.append(f'{folder_name} Avg SNR')
    
    for col, header in enumerate(summary_headers):
        worksheet.write(row, col, header)
    row += 1
    
    # Calculate and write summary statistics
    for band_name, freq_min, freq_max in frequency_bands:
        worksheet.write(row, 0, band_name)
        
        for col_idx, folder_name in enumerate(sorted_folders):
            if folder_name in folder_snr_data:
                # Collect all SNR values for this band in this folder
                band_snr_values = []
                for key, snr_value in folder_snr_data[folder_name].items():
                    if key[1] == band_name:  # key[1] is the band_name
                        band_snr_values.append(snr_value)
                
                if band_snr_values:
                    avg_snr = np.mean(band_snr_values)
                    worksheet.write(row, 1 + col_idx, round(avg_snr, 3))
                else:
                    worksheet.write(row, 1 + col_idx, 'N/A')
            else:
                worksheet.write(row, 1 + col_idx, 'No Data')
        
        row += 1

def create_overall_summary_statistics(workbook, all_analysis_data):
    """Create overall summary statistics worksheet"""
    
    worksheet = workbook.add_worksheet('Overall_Summary')
    
    # Group data by folder
    folder_stats = {}
    for data in all_analysis_data:
        folder = data['folder']
        if folder not in folder_stats:
            folder_stats[folder] = {'total_files': 0, 'noleak_files': 0, 'leak_files': 0, 'distances': set()}
        
        folder_stats[folder]['total_files'] += 1
        
        if data['is_noleak']:
            folder_stats[folder]['noleak_files'] += 1
        else:
            folder_stats[folder]['leak_files'] += 1
        
        # Extract distance
        distance_match = re.search(r'(\d+)m', data['filename'])
        if distance_match:
            folder_stats[folder]['distances'].add(distance_match.group(1))
    
    # Write headers
    headers = ['Folder', 'Total Files', 'NoLeak Files', 'Leak Files', 'Distances Tested']
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)
    
    row = 1
    
    # Write folder statistics
    for folder_name, stats in folder_stats.items():
        worksheet.write(row, 0, folder_name)
        worksheet.write(row, 1, stats['total_files'])
        worksheet.write(row, 2, stats['noleak_files'])
        worksheet.write(row, 3, stats['leak_files'])
        distances_str = ', '.join(sorted(stats['distances'], key=int)) + 'm'
        worksheet.write(row, 4, distances_str)
        row += 1
    
    # Add totals
    row += 1
    worksheet.write(row, 0, 'TOTALS:')
    worksheet.write(row, 1, sum(stats['total_files'] for stats in folder_stats.values()))
    worksheet.write(row, 2, sum(stats['noleak_files'] for stats in folder_stats.values()))
    worksheet.write(row, 3, sum(stats['leak_files'] for stats in folder_stats.values()))
    
    # All unique distances
    all_distances = set()
    for stats in folder_stats.values():
        all_distances.update(stats['distances'])
    distances_str = ', '.join(sorted(all_distances, key=int)) + 'm'
    worksheet.write(row, 4, distances_str)

def create_snr_frequency_bands_plot(workbook, all_analysis_data):
    """Create SNR vs Frequency Bands plot worksheet"""
    
    # Create worksheet
    plot_worksheet = workbook.add_worksheet('SNR_Frequency_Bands_Plot')
    
    # Define frequency bands for analysis (same as leak detection)
    frequency_bands = [
        # Very low frequency bands (1-100 Hz) - 10 bands
        ('Ultra-Low (1-10Hz)', 1, 10),
        ('Very Low (10-20Hz)', 10, 20),
        ('Low Structural (20-30Hz)', 20, 30),
        ('Mechanical (30-40Hz)', 30, 40),
        ('Power Harmonics (40-50Hz)', 40, 50),
        ('Power Frequency (50-60Hz)', 50, 60),
        ('Post-Power (60-70Hz)', 60, 70),
        ('Low Acoustic (70-80Hz)', 70, 80),
        ('Pre-Acoustic (80-90Hz)', 80, 90),
        ('Low Acoustic Transition (90-100Hz)', 90, 100),
        # Refined frequency bands
        ('Low Structural 1 (100-150Hz)', 100, 150),
        ('Low Structural 2 (150-200Hz)', 150, 200),
        ('Low Structural 3 (200-250Hz)', 200, 250),
        ('Low Structural 4 (250-300Hz)', 250, 300),
        ('Low Structural 5 (300-350Hz)', 300, 350),
        ('Low Structural 6 (350-400Hz)', 350, 400),
        ('Low Structural 7 (400-450Hz)', 400, 450),
        ('Low Structural 8 (450-500Hz)', 450, 500),
        ('Mid Frequency 1 (500-1500Hz)', 500, 1500),
        ('Mid Frequency 2 (1500-2000Hz)', 1500, 2000),
        ('High Frequency (2000-8000Hz)', 2000, 8000),
        ('Ultrasonic (8000-20000Hz)', 8000, 20000)
    ]
    
    # Group data by folder
    folder_groups = {}
    for data in all_analysis_data:
        folder = data['folder']
        if folder not in folder_groups:
            folder_groups[folder] = {'noleak': [], 'leak': []}
        
        if data['is_noleak']:
            folder_groups[folder]['noleak'].append(data)
        else:
            folder_groups[folder]['leak'].append(data)
    
    # Get sorted folder names
    def extract_lh_number(folder_name):
        lh_match = re.search(r'(\d+)lh', folder_name, re.IGNORECASE)
        if lh_match:
            return int(lh_match.group(1))
        else:
            return float('inf')
    
    sorted_folders = sorted(folder_groups.keys(), key=lambda x: (extract_lh_number(x), x))
    
    # Collect unique measurement types
    unique_measurements = set()
    
    for folder_name in sorted_folders:
        folder_data = folder_groups[folder_name]
        if folder_data['noleak'] and folder_data['leak']:
            leak_measurements = folder_data['leak']
            for measurement in leak_measurements:
                # Extract measurement name (same logic as in FFT bands comparison)
                filename = measurement['filename']
                
                distance_match = re.search(r'(\d+)m(?:[_\s]?([A-Za-z]+))?', filename)
                if distance_match:
                    distance_str = f"{distance_match.group(1)}m"
                    additional_word = distance_match.group(2)
                    
                    if additional_word and additional_word.lower() not in ['leak', 'noleak']:
                        distance_part = f"{distance_str}_{additional_word}"
                    else:
                        distance_part = distance_str
                else:
                    distance_part = "Unknown"
                
                if '_leak' in filename.lower():
                    leak_designation = "Leak"
                elif '_noleak' in filename.lower():
                    leak_designation = "NoLeak"
                else:
                    leak_designation = "Unknown"
                
                worksheet_name = f"{distance_part}_{leak_designation}"
                distance = int(distance_match.group(1)) if distance_match else 0
                
                unique_measurements.add((worksheet_name, distance))
    
    # Sort measurements by distance
    sorted_measurements = sorted(unique_measurements, key=lambda x: (x[1], x[0]))
    
    # Calculate SNR data for plotting
    plot_data = {}
    for folder_name in sorted_folders:
        folder_data = folder_groups[folder_name]
        if not folder_data['noleak'] or not folder_data['leak']:
            continue
        
        # Calculate folder-specific FFT noise floor
        folder_fft_noise = []
        for noleak_data in folder_data['noleak']:
            folder_fft_noise.append(noleak_data['fft_abs_min'])
        
        if folder_fft_noise:
            folder_noise_floor = np.mean(np.vstack(folder_fft_noise), axis=0)
            folder_noise_floor = np.maximum(folder_noise_floor, np.finfo(float).eps)
            
            # Process leak measurements
            leak_measurements = folder_data['leak']
            for measurement in leak_measurements:
                # Get measurement name
                filename = measurement['filename']
                
                distance_match = re.search(r'(\d+)m(?:[_\s]?([A-Za-z]+))?', filename)
                if distance_match:
                    distance_str = f"{distance_match.group(1)}m"
                    additional_word = distance_match.group(2)
                    
                    if additional_word and additional_word.lower() not in ['leak', 'noleak']:
                        distance_part = f"{distance_str}_{additional_word}"
                    else:
                        distance_part = distance_str
                else:
                    distance_part = "Unknown"
                
                worksheet_name = f"{distance_part}_Leak"
                
                # Calculate SNR for each frequency band
                freq = measurement['frequency']
                signal = measurement['fft_abs_min']
                snr_values = signal / folder_noise_floor
                
                measurement_key = f"{worksheet_name}_{folder_name}"
                if measurement_key not in plot_data:
                    plot_data[measurement_key] = {}
                
                for band_name, freq_min, freq_max in frequency_bands:
                    band_mask = (freq >= freq_min) & (freq <= freq_max)
                    
                    if np.any(band_mask):
                        band_snr = np.mean(snr_values[band_mask])
                        plot_data[measurement_key][band_name] = band_snr
    
    # Prepare data for Excel chart
    # Write frequency band names in first column
    row = 0
    plot_worksheet.write(row, 0, 'Frequency Band')
    
    # Write measurement type headers
    col = 1
    measurement_columns = {}
    for measurement_name, distance in sorted_measurements:
        for folder_name in sorted_folders:
            measurement_key = f"{measurement_name}_{folder_name}"
            if measurement_key in plot_data:
                header = f"{measurement_name} ({folder_name})"
                plot_worksheet.write(row, col, header)
                measurement_columns[measurement_key] = col
                col += 1
    
    # Write frequency band data
    row = 1
    for band_name, freq_min, freq_max in frequency_bands:
        plot_worksheet.write(row, 0, band_name)
        
        for measurement_key, col_idx in measurement_columns.items():
            if band_name in plot_data[measurement_key]:
                snr_value = plot_data[measurement_key][band_name]
                plot_worksheet.write(row, col_idx, round(snr_value, 3))
            else:
                plot_worksheet.write(row, col_idx, 0)
        
        row += 1
    
    # Create the chart
    chart = workbook.add_chart({'type': 'line'})
    
    # Define colors (same as existing plots)
    distinguishable_colors = ['#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF', '#FF0080', '#00FFFF', '#FFFF00', 
                             '#800000', '#000080', '#008000', '#804000', '#400080', '#800040', '#008080', '#808000']
    
    # Add series for each measurement type
    color_index = 0
    for measurement_key, col_idx in measurement_columns.items():
        # Extract measurement display name
        parts = measurement_key.split('_')
        if len(parts) >= 3:
            display_name = f"{parts[0]}_{parts[1]} ({parts[2]})"
        else:
            display_name = measurement_key
        
        chart.add_series({
            'name': display_name,
            'categories': ['SNR_Frequency_Bands_Plot', 1, 0, len(frequency_bands), 0],  # Frequency band names
            'values': ['SNR_Frequency_Bands_Plot', 1, col_idx, len(frequency_bands), col_idx],  # SNR values
            'line': {
                'color': distinguishable_colors[color_index % len(distinguishable_colors)],
                'width': 2
            },
            'marker': {
                'type': 'circle',
                'size': 5,
                'border': {'color': distinguishable_colors[color_index % len(distinguishable_colors)]},
                'fill': {'color': distinguishable_colors[color_index % len(distinguishable_colors)]}
            }
        })
        color_index += 1
    
    # Configure chart appearance (similar to existing plots)
    chart.set_title({'name': 'SNR vs Frequency Bands by Measurement Type', 'name_font': {'size': 12}})
    chart.set_x_axis({
        'name': 'Frequency Bands',
        'label_position': 'low',
        'name_font': {'size': 12},
        'num_font': {'size': 10},
        'text_axis': True  # Treat as text categories instead of numeric
    })
    chart.set_y_axis({
        'name': 'SNR (Linear)',
        'label_position': 'low',
        'name_layout': {'x': 0.02, 'y': 0.5},
        'name_font': {'size': 12},
        'num_font': {'size': 12}
    })
    chart.set_size({'width': 1400, 'height': 700})
    chart.set_plotarea({'layout': {'x': 0.15, 'y': 0.15, 'width': 0.75, 'height': 0.70}})
    chart.set_legend({'font': {'size': 12}})
    
    # Insert chart into worksheet
    plot_worksheet.insert_chart('A' + str(len(frequency_bands) + 3), chart)

def create_file_specific_fft_bands_snr_comparison(workbook, all_analysis_data):
    """Create file-specific FFT bands SNR comparison table across all folders"""
    
    # Create worksheet
    worksheet = workbook.add_worksheet('File_Specific_FFT_Bands_SNR')
    
    # Define frequency bands for analysis (same as leak detection)
    frequency_bands = [
        # Very low frequency bands (1-100 Hz) - 10 bands
        ('Ultra-Low (1-10Hz)', 1, 10),       # Ultra-low frequency structural
        ('Very Low (10-20Hz)', 10, 20),      # Very low frequency vibrations
        ('Low Structural (20-30Hz)', 20, 30),      # Low frequency structural response
        ('Mechanical (30-40Hz)', 30, 40),      # Mechanical vibrations
        ('Power Harmonics (40-50Hz)', 40, 50),      # Power line and mechanical harmonics
        ('Power Frequency (50-60Hz)', 50, 60),      # Power frequency range
        ('Post-Power (60-70Hz)', 60, 70),      # Post-power frequency
        ('Low Acoustic (70-80Hz)', 70, 80),      # Low acoustic range
        ('Pre-Acoustic (80-90Hz)', 80, 90),      # Pre-acoustic range
        ('Low Acoustic Transition (90-100Hz)', 90, 100),     # Low acoustic transition
        # Refined frequency bands
        # 100-500 Hz range in 50 Hz increments
        ('Low Structural 1 (100-150Hz)', 100, 150),    # Low structural range 1
        ('Low Structural 2 (150-200Hz)', 150, 200),    # Low structural range 2
        ('Low Structural 3 (200-250Hz)', 200, 250),    # Low structural range 3
        ('Low Structural 4 (250-300Hz)', 250, 300),    # Low structural range 4
        ('Low Structural 5 (300-350Hz)', 300, 350),    # Low structural range 5
        ('Low Structural 6 (350-400Hz)', 350, 400),    # Low structural range 6
        ('Low Structural 7 (400-450Hz)', 400, 450),    # Low structural range 7
        ('Low Structural 8 (450-500Hz)', 450, 500),    # Low structural range 8
        # 500-2000 Hz range in 1000 Hz jumps
        ('Mid Frequency 1 (500-1500Hz)', 500, 1500),   # Mid frequency acoustic emissions 1
        ('Mid Frequency 2 (1500-2000Hz)', 1500, 2000),  # Mid frequency acoustic emissions 2
        # Original higher frequency bands
        ('High Frequency (2000-8000Hz)', 2000, 8000),  # High frequency turbulence
        ('Ultrasonic (8000-20000Hz)', 8000, 20000)  # Ultrasonic range
    ]
    
    # Group data by folder and file
    folder_file_groups = {}
    for data in all_analysis_data:
        folder = data['folder']
        # Extract base filename without _leak/_noleak suffix
        base_filename = re.sub(r'_(leak|noleak)$', '', data['filename'])
        
        if folder not in folder_file_groups:
            folder_file_groups[folder] = {}
        
        if base_filename not in folder_file_groups[folder]:
            folder_file_groups[folder][base_filename] = {'noleak': [], 'leak': []}
        
        if data['is_noleak']:
            folder_file_groups[folder][base_filename]['noleak'].append(data)
        else:
            folder_file_groups[folder][base_filename]['leak'].append(data)
    
    # Get sorted folder names for column headers
    def extract_lh_number(folder_name):
        lh_match = re.search(r'(\d+)lh', folder_name, re.IGNORECASE)
        if lh_match:
            return int(lh_match.group(1))
        else:
            return float('inf')
    
    sorted_folders = sorted(folder_file_groups.keys(), key=lambda x: (extract_lh_number(x), x))
    
    # Create a structure to hold all measurement/frequency band combinations
    measurement_band_combinations = []
    unique_measurements = set()
    
    # Collect all unique measurements from all folders
    for folder_name in sorted_folders:
        folder_data = folder_file_groups[folder_name]
        for base_filename, file_data in folder_data.items():
            if file_data['noleak'] and file_data['leak']:
                leak_measurements = file_data['leak']
                for measurement in leak_measurements:
                    # Extract distance from filename
                    distance_match = re.search(r'(\d+)m', measurement['filename'])
                    distance = int(distance_match.group(1)) if distance_match else 0
                    
                    # Create worksheet tab name (same logic as in individual folder processing)
                    filename = measurement['filename']
                    
                    # Extract distance and any additional word after it
                    distance_match = re.search(r'(\d+)m(?:[_\s]?([A-Za-z]+))?', filename)
                    if distance_match:
                        distance_str = f"{distance_match.group(1)}m"
                        additional_word = distance_match.group(2)  # The word after 'm' (if any)
                        
                        # Build distance part with additional word if present
                        if additional_word and additional_word.lower() not in ['leak', 'noleak']:
                            distance_part = f"{distance_str}_{additional_word}"
                        else:
                            distance_part = distance_str
                    else:
                        distance_part = "Unknown"
                    
                    # Determine leak/noleak designation
                    if '_leak' in filename.lower():
                        leak_designation = "Leak"
                    elif '_noleak' in filename.lower():
                        leak_designation = "NoLeak"
                    else:
                        leak_designation = "Unknown"
                    
                    # Create worksheet name: "5m_sensor_Leak", "10m_test_NoLeak", etc.
                    worksheet_name = f"{distance_part}_{leak_designation}"
                    
                    # Add to unique measurements set using worksheet name
                    measurement_key = (worksheet_name, distance)
                    unique_measurements.add(measurement_key)
    
    # Create combinations for all unique measurements and all frequency bands
    # Sort measurements by distance first, then by worksheet name
    sorted_measurements = sorted(unique_measurements, key=lambda x: (x[1], x[0]))
    
    for worksheet_name, distance in sorted_measurements:
        for band_name, freq_min, freq_max in frequency_bands:
            combination = (worksheet_name, distance, band_name)
            measurement_band_combinations.append(combination)
    
    # Write headers - basic info columns followed by folder SNR columns
    headers = ['Measurement', 'Distance (m)', 'Frequency Band']
    for folder_name in sorted_folders:
        headers.append(f'{folder_name} File-Specific SNR')
    
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)
    
    # Calculate file-specific SNR values for each folder
    folder_snr_data = {}
    for folder_name in sorted_folders:
        folder_data = folder_file_groups[folder_name]
        folder_snr_data[folder_name] = {}
        
        for base_filename, file_data in folder_data.items():
            if not file_data['noleak'] or not file_data['leak']:
                continue
            
            # Calculate file-specific FFT noise floor (only from this file's NoLeak data)
            file_fft_noise = []
            for noleak_data in file_data['noleak']:
                file_fft_noise.append(noleak_data['fft_abs_min'])
            
            if file_fft_noise:
                # Use improved file-specific noise floor estimation
                file_noise_floor = improved_noise_floor_estimation(
                    file_fft_noise,
                    method='median',  # Use median for file-specific (smaller sample sizes)
                    min_noise_threshold=1e-12
                )
                
                # Process leak measurements
                leak_measurements = file_data['leak']
                for measurement in leak_measurements:
                    # Calculate robust SNR for each frequency band
                    freq = measurement['frequency']
                    signal = measurement['fft_abs_min']
                    snr_values = calculate_robust_snr(
                        signal, 
                        file_noise_floor,
                        method='clipped_linear',
                        max_snr_db=60,
                        min_snr_db=-40
                    )
                    
                    # Create worksheet tab name (same logic as in individual folder processing)
                    filename = measurement['filename']
                    
                    # Extract distance and any additional word after it
                    distance_match = re.search(r'(\d+)m(?:[_\s]?([A-Za-z]+))?', filename)
                    if distance_match:
                        distance_str = f"{distance_match.group(1)}m"
                        additional_word = distance_match.group(2)  # The word after 'm' (if any)
                        
                        # Build distance part with additional word if present
                        if additional_word and additional_word.lower() not in ['leak', 'noleak']:
                            distance_part = f"{distance_str}_{additional_word}"
                        else:
                            distance_part = distance_str
                    else:
                        distance_part = "Unknown"
                    
                    # Determine leak/noleak designation
                    if '_leak' in filename.lower():
                        leak_designation = "Leak"
                    elif '_noleak' in filename.lower():
                        leak_designation = "NoLeak"
                    else:
                        leak_designation = "Unknown"
                    
                    # Create worksheet name: "5m_sensor_Leak", "10m_test_NoLeak", etc.
                    worksheet_name = f"{distance_part}_{leak_designation}"
                    
                    for band_name, freq_min, freq_max in frequency_bands:
                        # Find frequency indices for this band
                        band_mask = (freq >= freq_min) & (freq <= freq_max)
                        
                        if np.any(band_mask):
                            band_snr = np.mean(snr_values[band_mask])
                            key = (worksheet_name, band_name)
                            folder_snr_data[folder_name][key] = band_snr
    
    # Write data rows
    row = 1
    for worksheet_name, distance, band_name in measurement_band_combinations:
        # Write basic info
        worksheet.write(row, 0, worksheet_name)
        worksheet.write(row, 1, distance)
        worksheet.write(row, 2, band_name)
        
        # Write SNR values for each folder
        for col_idx, folder_name in enumerate(sorted_folders):
            if folder_name in folder_snr_data:
                key = (worksheet_name, band_name)
                if key in folder_snr_data[folder_name]:
                    snr_value = folder_snr_data[folder_name][key]
                    worksheet.write(row, 3 + col_idx, round(snr_value, 3))
                else:
                    worksheet.write(row, 3 + col_idx, 'N/A')
            else:
                worksheet.write(row, 3 + col_idx, 'No Data')
        
        row += 1
    
    # Add summary statistics section
    row += 2
    worksheet.write(row, 0, 'SUMMARY STATISTICS (File-Specific)')
    row += 2
    
    # Calculate folder averages for each frequency band
    worksheet.write(row, 0, 'Average File-Specific SNR by Frequency Band:')
    row += 1
    
    # Write summary headers
    summary_headers = ['Frequency Band']
    for folder_name in sorted_folders:
        summary_headers.append(f'{folder_name} Avg File-Specific SNR')
    
    for col, header in enumerate(summary_headers):
        worksheet.write(row, col, header)
    row += 1
    
    # Calculate and write summary statistics
    for band_name, freq_min, freq_max in frequency_bands:
        worksheet.write(row, 0, band_name)
        
        for col_idx, folder_name in enumerate(sorted_folders):
            if folder_name in folder_snr_data:
                # Collect all SNR values for this band in this folder
                band_snr_values = []
                for key, snr_value in folder_snr_data[folder_name].items():
                    if key[1] == band_name:  # key[1] is the band_name
                        band_snr_values.append(snr_value)
                
                if band_snr_values:
                    avg_snr = np.mean(band_snr_values)
                    worksheet.write(row, 1 + col_idx, round(avg_snr, 3))
                else:
                    worksheet.write(row, 1 + col_idx, 'N/A')
            else:
                worksheet.write(row, 1 + col_idx, 'No Data')
        
        row += 1

def create_file_specific_overall_summary_statistics(workbook, all_analysis_data):
    """Create file-specific overall summary statistics worksheet"""
    
    worksheet = workbook.add_worksheet('File_Specific_Overall_Summary')
    
    # Group data by folder and file
    folder_file_stats = {}
    for data in all_analysis_data:
        folder = data['folder']
        # Extract base filename without _leak/_noleak suffix
        base_filename = re.sub(r'_(leak|noleak)$', '', data['filename'])
        
        if folder not in folder_file_stats:
            folder_file_stats[folder] = {}
        
        if base_filename not in folder_file_stats[folder]:
            folder_file_stats[folder][base_filename] = {
                'total_files': 0, 'noleak_files': 0, 'leak_files': 0, 'distances': set()
            }
        
        folder_file_stats[folder][base_filename]['total_files'] += 1
        
        if data['is_noleak']:
            folder_file_stats[folder][base_filename]['noleak_files'] += 1
        else:
            folder_file_stats[folder][base_filename]['leak_files'] += 1
        
        # Extract distance from filename
        distance_match = re.search(r'(\d+)m', data['filename'])
        if distance_match:
            distance = distance_match.group(1)
            folder_file_stats[folder][base_filename]['distances'].add(distance)
    
    # Sort folders
    def extract_lh_number(folder_name):
        lh_match = re.search(r'(\d+)lh', folder_name, re.IGNORECASE)
        if lh_match:
            return int(lh_match.group(1))
        else:
            return float('inf')
    
    sorted_folders = sorted(folder_file_stats.keys(), key=lambda x: (extract_lh_number(x), x))
    
    # Write headers
    headers = ['Folder', 'Base Filename', 'Total Files', 'NoLeak Files', 'Leak Files', 'Distances']
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)
    
    # Write data
    row = 1
    for folder in sorted_folders:
        file_stats = folder_file_stats[folder]
        for base_filename, stats in sorted(file_stats.items()):
            worksheet.write(row, 0, folder)
            worksheet.write(row, 1, base_filename)
            worksheet.write(row, 2, stats['total_files'])
            worksheet.write(row, 3, stats['noleak_files'])
            worksheet.write(row, 4, stats['leak_files'])
            
            # Convert distances to sorted string
            all_distances = list(stats['distances'])
            distances_str = ', '.join(sorted(all_distances, key=int)) + 'm'
            worksheet.write(row, 5, distances_str)
            
            row += 1
    
    # Add summary section
    row += 2
    worksheet.write(row, 0, 'SUMMARY BY FOLDER (File-Specific Analysis)')
    row += 2
    
    # Summary headers
    summary_headers = ['Folder', 'Total Base Files', 'Total File Pairs', 'Avg Files per Base']
    for col, header in enumerate(summary_headers):
        worksheet.write(row, col, header)
    row += 1
    
    # Calculate and write summary for each folder
    for folder in sorted_folders:
        file_stats = folder_file_stats[folder]
        total_base_files = len(file_stats)
        total_file_pairs = sum(stats['total_files'] for stats in file_stats.values())
        avg_files_per_base = total_file_pairs / total_base_files if total_base_files > 0 else 0
        
        worksheet.write(row, 0, folder)
        worksheet.write(row, 1, total_base_files)
        worksheet.write(row, 2, total_file_pairs)
        worksheet.write(row, 3, round(avg_files_per_base, 1))
        
        row += 1

def create_file_specific_snr_frequency_bands_plot(workbook, all_analysis_data):
    """Create file-specific SNR vs Frequency Bands plot worksheet"""
    
    # Create worksheet
    plot_worksheet = workbook.add_worksheet('File_Specific_SNR_Plot')
    
    # Define frequency bands for analysis (same as leak detection)
    frequency_bands = [
        # Very low frequency bands (1-100 Hz) - 10 bands
        ('Ultra-Low (1-10Hz)', 1, 10),
        ('Very Low (10-20Hz)', 10, 20),
        ('Low Structural (20-30Hz)', 20, 30),
        ('Mechanical (30-40Hz)', 30, 40),
        ('Power Harmonics (40-50Hz)', 40, 50),
        ('Power Frequency (50-60Hz)', 50, 60),
        ('Post-Power (60-70Hz)', 60, 70),
        ('Low Acoustic (70-80Hz)', 70, 80),
        ('Pre-Acoustic (80-90Hz)', 80, 90),
        ('Low Acoustic Transition (90-100Hz)', 90, 100),
        # Refined frequency bands
        ('Low Structural 1 (100-150Hz)', 100, 150),
        ('Low Structural 2 (150-200Hz)', 150, 200),
        ('Low Structural 3 (200-250Hz)', 200, 250),
        ('Low Structural 4 (250-300Hz)', 250, 300),
        ('Low Structural 5 (300-350Hz)', 300, 350),
        ('Low Structural 6 (350-400Hz)', 350, 400),
        ('Low Structural 7 (400-450Hz)', 400, 450),
        ('Low Structural 8 (450-500Hz)', 450, 500),
        ('Mid Frequency 1 (500-1500Hz)', 500, 1500),
        ('Mid Frequency 2 (1500-2000Hz)', 1500, 2000),
        ('High Frequency (2000-8000Hz)', 2000, 8000),
        ('Ultrasonic (8000-20000Hz)', 8000, 20000)
    ]
    
    # Group data by folder and file
    folder_file_groups = {}
    for data in all_analysis_data:
        folder = data['folder']
        # Extract base filename without _leak/_noleak suffix
        base_filename = re.sub(r'_(leak|noleak)$', '', data['filename'])
        
        if folder not in folder_file_groups:
            folder_file_groups[folder] = {}
        
        if base_filename not in folder_file_groups[folder]:
            folder_file_groups[folder][base_filename] = {'noleak': [], 'leak': []}
        
        if data['is_noleak']:
            folder_file_groups[folder][base_filename]['noleak'].append(data)
        else:
            folder_file_groups[folder][base_filename]['leak'].append(data)
    
    # Get sorted folder names
    def extract_lh_number(folder_name):
        lh_match = re.search(r'(\d+)lh', folder_name, re.IGNORECASE)
        if lh_match:
            return int(lh_match.group(1))
        else:
            return float('inf')
    
    sorted_folders = sorted(folder_file_groups.keys(), key=lambda x: (extract_lh_number(x), x))
    
    # Collect unique measurement types
    unique_measurements = set()
    
    for folder_name in sorted_folders:
        folder_data = folder_file_groups[folder_name]
        for base_filename, file_data in folder_data.items():
            if file_data['noleak'] and file_data['leak']:
                leak_measurements = file_data['leak']
                for measurement in leak_measurements:
                    # Extract measurement name (same logic as in FFT bands comparison)
                    filename = measurement['filename']
                    
                    distance_match = re.search(r'(\d+)m(?:[_\s]?([A-Za-z]+))?', filename)
                    if distance_match:
                        distance_str = f"{distance_match.group(1)}m"
                        additional_word = distance_match.group(2)
                        
                        if additional_word and additional_word.lower() not in ['leak', 'noleak']:
                            distance_part = f"{distance_str}_{additional_word}"
                        else:
                            distance_part = distance_str
                    else:
                        distance_part = "Unknown"
                    
                    if '_leak' in filename.lower():
                        leak_designation = "Leak"
                    elif '_noleak' in filename.lower():
                        leak_designation = "NoLeak"
                    else:
                        leak_designation = "Unknown"
                    
                    worksheet_name = f"{distance_part}_{leak_designation}"
                    distance = int(distance_match.group(1)) if distance_match else 0
                    
                    unique_measurements.add((worksheet_name, distance))
    
    # Sort measurements by distance
    sorted_measurements = sorted(unique_measurements, key=lambda x: (x[1], x[0]))
    
    # Calculate file-specific SNR data for plotting
    plot_data = {}
    for folder_name in sorted_folders:
        folder_data = folder_file_groups[folder_name]
        
        for base_filename, file_data in folder_data.items():
            if not file_data['noleak'] or not file_data['leak']:
                continue
            
            # Calculate file-specific FFT noise floor (only from this file's NoLeak data)
            file_fft_noise = []
            for noleak_data in file_data['noleak']:
                file_fft_noise.append(noleak_data['fft_abs_min'])
            
            if file_fft_noise:
                # Use improved file-specific noise floor estimation
                file_noise_floor = improved_noise_floor_estimation(
                    file_fft_noise,
                    method='median',  # Use median for file-specific (smaller sample sizes)
                    min_noise_threshold=1e-12
                )
                
                # Process leak measurements
                leak_measurements = file_data['leak']
                for measurement in leak_measurements:
                    # Get measurement name
                    filename = measurement['filename']
                    
                    distance_match = re.search(r'(\d+)m(?:[_\s]?([A-Za-z]+))?', filename)
                    if distance_match:
                        distance_str = f"{distance_match.group(1)}m"
                        additional_word = distance_match.group(2)
                        
                        if additional_word and additional_word.lower() not in ['leak', 'noleak']:
                            distance_part = f"{distance_str}_{additional_word}"
                        else:
                            distance_part = distance_str
                    else:
                        distance_part = "Unknown"
                    
                    worksheet_name = f"{distance_part}_Leak"
                    
                    # Calculate robust SNR for each frequency band
                    freq = measurement['frequency']
                    signal = measurement['fft_abs_min']
                    snr_values = calculate_robust_snr(
                        signal, 
                        file_noise_floor,
                        method='clipped_linear',
                        max_snr_db=60,
                        min_snr_db=-40
                    )
                    
                    measurement_key = f"{worksheet_name}_{folder_name}_{base_filename}"
                    if measurement_key not in plot_data:
                        plot_data[measurement_key] = {}
                    
                    for band_name, freq_min, freq_max in frequency_bands:
                        band_mask = (freq >= freq_min) & (freq <= freq_max)
                        
                        if np.any(band_mask):
                            band_snr = np.mean(snr_values[band_mask])
                            plot_data[measurement_key][band_name] = band_snr
    
    # Prepare data for Excel chart
    # Write frequency band names in first column
    row = 0
    plot_worksheet.write(row, 0, 'Frequency Band')
    
    # Write measurement type headers
    col = 1
    measurement_columns = {}
    for measurement_name, distance in sorted_measurements:
        for folder_name in sorted_folders:
            folder_data = folder_file_groups[folder_name]
            for base_filename in folder_data.keys():
                measurement_key = f"{measurement_name}_{folder_name}_{base_filename}"
                if measurement_key in plot_data:
                    header = f"{measurement_name} ({folder_name}_{base_filename})"
                    plot_worksheet.write(row, col, header)
                    measurement_columns[measurement_key] = col
                    col += 1
    
    # Write frequency band data
    row = 1
    for band_name, freq_min, freq_max in frequency_bands:
        plot_worksheet.write(row, 0, band_name)
        
        for measurement_key, col_idx in measurement_columns.items():
            if band_name in plot_data[measurement_key]:
                snr_value = plot_data[measurement_key][band_name]
                plot_worksheet.write(row, col_idx, round(snr_value, 3))
            else:
                plot_worksheet.write(row, col_idx, 0)
        
        row += 1
    
    # Create the chart
    chart = workbook.add_chart({'type': 'line'})
    
    # Define colors (same as existing plots)
    distinguishable_colors = ['#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF', '#FF0080', '#00FFFF', '#FFFF00', 
                             '#800000', '#000080', '#008000', '#804000', '#400080', '#800040', '#008080', '#808000']
    
    # Add series for each measurement type
    color_index = 0
    for measurement_key, col_idx in measurement_columns.items():
        # Extract measurement display name
        parts = measurement_key.split('_')
        if len(parts) >= 4:
            display_name = f"{parts[0]}_{parts[1]} ({parts[2]}_{parts[3]})"
        elif len(parts) >= 3:
            display_name = f"{parts[0]}_{parts[1]} ({parts[2]})"
        else:
            display_name = measurement_key
        
        chart.add_series({
            'name': display_name,
            'categories': ['File_Specific_SNR_Plot', 1, 0, len(frequency_bands), 0],  # Frequency band names
            'values': ['File_Specific_SNR_Plot', 1, col_idx, len(frequency_bands), col_idx],  # SNR values
            'line': {
                'color': distinguishable_colors[color_index % len(distinguishable_colors)],
                'width': 2
            },
            'marker': {
                'type': 'circle',
                'size': 5,
                'border': {'color': distinguishable_colors[color_index % len(distinguishable_colors)]},
                'fill': {'color': distinguishable_colors[color_index % len(distinguishable_colors)]}
            }
        })
        color_index += 1
    
    # Configure chart appearance (similar to existing plots)
    chart.set_title({'name': 'File-Specific SNR vs Frequency Bands by Measurement Type', 'name_font': {'size': 12}})
    chart.set_x_axis({
        'name': 'Frequency Bands',
        'label_position': 'low',
        'name_font': {'size': 12},
        'num_font': {'size': 10},
        'text_axis': True  # Treat as text categories instead of numeric
    })
    chart.set_y_axis({
        'name': 'File-Specific SNR (Linear)',
        'label_position': 'low',
        'name_layout': {'x': 0.02, 'y': 0.5},
        'name_font': {'size': 12},
        'num_font': {'size': 12}
    })
    chart.set_size({'width': 1400, 'height': 700})
    chart.set_plotarea({'layout': {'x': 0.15, 'y': 0.15, 'width': 0.75, 'height': 0.70}})
    chart.set_legend({'font': {'size': 12}})
    
    # Insert chart into worksheet
    plot_worksheet.insert_chart('A' + str(len(frequency_bands) + 3), chart)

# Create the summary comparison file
create_summary_comparison(all_folders_data, path)

print(f"\n" + "="*50)
print(f"🎵 ANALYSIS COMPLETE")
print(f"="*50)
print(f"- Total subfolders scanned: {len(subfolders)}")
print(f"- Total WAV files with distance pattern processed: {total_wav_files}")
print(f"- Total segments extracted: {total_segments}")
distance_pattern = r'\d+m'
print(f"- Analysis files created: {len([f for f in subfolders if len([x for x in os.listdir(os.path.join(path, f)) if x.lower().endswith('.wav') and re.search(distance_pattern, x)]) > 0])}")
print(f"- Summary comparison file created: summary_comparison.xlsx")
print(f"\nNote: Only WAV files containing distance pattern (number + 'm') were processed.")
print(f"Each folder with valid files now contains its own analysis summary file.")
print(f"The summary_comparison.xlsx file contains cross-folder FFT bands SNR analysis.")

print('OK')
