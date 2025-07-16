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
path = r'D:\OneDrive - Arad Technologies Ltd\ARAD_Projects\ALD\tests\audio_data'

# Get all subfolders in the main path
subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

ListFileNames = []
ListDataFrames = []

n_samples = 131072  # 2^n  -> 131072 (2^17)
n_AVG = 7

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
        snr_dB = np.zeros_like(psd_AVG)  # Placeholder, will be calculated later
        snr_distance_dB = np.zeros_like(psd_AVG)  # Placeholder for distance-specific SNR
        
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
        headers = ['Time', 'Data', 'Frequency', 'FFT_Real', 'FFT_Imag', 'FFT_Abs', 'FFT_MIN_Real', 'FFT_MIN_Imag', 'FFT_MIN_Abs', 'PSD', 'SNR_dB', 'SNR_Distance_dB']
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
            summaryWorksheet.write(i+1,10,sanitize_excel_value(snr_dB[i]))
            summaryWorksheet.write(i+1,11,sanitize_excel_value(snr_distance_dB[i]))
            
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
        for data in analysis_data:
            if data['is_noleak']:
                noleak_psd_data.append(data['psd_avg'])
        
        if noleak_psd_data:
            # Create global noise floor baseline
            noise_floor = np.mean(np.vstack(noleak_psd_data), axis=0)
            noise_floor = np.maximum(noise_floor, np.finfo(float).eps)  # Avoid division by zero
            
            # Update SNR data for all measurements
            for data in analysis_data:
                # Calculate SNR in dB for all measurements (including NoLeak)
                snr_dB = 10 * np.log10(data['psd_avg'] / noise_floor)
                
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
                            for j in range(len(snr_dB)):
                                ws.write(j+1, 10, sanitize_excel_value(snr_dB[j]))
                            break

    # Perform leak detection analysis
    leak_detection_results = analyze_leak_detection(analysis_data)
    
    def analyze_leak_detection_distance_specific(analysis_data):
        """Analyze leak detection using distance-specific noise floors"""
        
        # Group data by distance
        distance_groups = {}
        
        for data in analysis_data:
            # Extract distance from filename
            match = re.search(r'(\d+)m', data['filename'])
            if match:
                distance = int(match.group(1))
                if distance not in distance_groups:
                    distance_groups[distance] = {'noleak': [], 'potential_leaks': []}
                
                if data['is_noleak']:
                    distance_groups[distance]['noleak'].append(data)
                else:
                    distance_groups[distance]['potential_leaks'].append(data)
        
        # Analyze each distance group
        results = {}
        
        for distance, group in distance_groups.items():
            if not group['noleak']:
                # No NoLeak baseline for this distance
                for leak_data in group['potential_leaks']:
                    results[leak_data['filename']] = {
                        'error': f'No NoLeak baseline available for {distance}m distance',
                        'distance': distance,
                        'baseline_available': False
                    }
                continue
            
            # Create distance-specific baseline
            noleak_baseline = np.vstack([data['psd_avg'] for data in group['noleak']])
            
            # Analyze potential leaks at this distance
            for leak_data in group['potential_leaks']:
                detection_result = calculate_leak_detection_score(
                    leak_data['frequency'], leak_data['psd_avg'], noleak_baseline
                )
                
                # Add distance-specific information
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
                    'values': [series_info['sheet_name'], 1, 10, series_info['data_points'], 10],     # SNR_dB column (column K)
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
                    'values': [series_info['sheet_name'], 1, 10, series_info['data_points'], 10],     # SNR_dB column (column K)
                    'line': {'width': 1}
                }
                # Apply grey color
                grey_color = grey_colors[grey_index % len(grey_colors)]
                series_config['line']['color'] = grey_color
                grey_index += 1
                chartSNR.add_series(series_config)
        
        # Configure SNR chart
        chartSNR.set_title({'name': 'Signal-to-Noise Ratio vs Frequency', 'name_font': {'size': 12}})
        chartSNR.set_x_axis({
            'name': 'Frequency (Hz)',
            'log_base': 10,
            'label_position': 'low',
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartSNR.set_y_axis({
            'name': 'SNR (dB)',
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

        # Create Distance-Specific SNR plot worksheet
        snr_distance_sheet_name = 'SNR_Distance_Specific'
        counter = 1
        while snr_distance_sheet_name in used_worksheet_names:
            snr_distance_sheet_name = f'SNR_Distance_Specific_{counter}'
            counter += 1
        used_worksheet_names.add(snr_distance_sheet_name)
        snrDistanceWorksheet = summaryWorkbook.add_worksheet(snr_distance_sheet_name)
        
        # Calculate distance-specific SNR data
        if analysis_data:
            # Group data by distance
            distance_groups = {}
            
            for data in analysis_data:
                # Extract distance from filename
                match = re.search(r'(\d+)m', data['filename'])
                if match:
                    distance = int(match.group(1))
                    if distance not in distance_groups:
                        distance_groups[distance] = {'noleak': [], 'all_measurements': []}
                    
                    distance_groups[distance]['all_measurements'].append(data)
                    if data['is_noleak']:
                        distance_groups[distance]['noleak'].append(data)
            
            # Calculate distance-specific SNR for each group
            for distance, group in distance_groups.items():
                if group['noleak']:
                    # Create distance-specific noise floor
                    distance_noise_floor = np.mean(np.vstack([data['psd_avg'] for data in group['noleak']]), axis=0)
                    distance_noise_floor = np.maximum(distance_noise_floor, np.finfo(float).eps)
                    
                    # Calculate SNR for all measurements at this distance
                    for data in group['all_measurements']:
                        # Calculate distance-specific SNR in dB
                        snr_distance_dB = 10 * np.log10(data['psd_avg'] / distance_noise_floor)
                        
                        # Update the worksheet with distance-specific SNR values
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
                                    # Update distance-specific SNR column (column L, index 11)
                                    for j in range(len(snr_distance_dB)):
                                        ws.write(j+1, 11, sanitize_excel_value(snr_distance_dB[j]))
                                    break
        
        # Create scatter chart with straight lines for distance-specific SNR
        chartSNRDist = summaryWorkbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
        
        # First add all non-NoLeak series (colorful lines in background)
        color_index = 0
        for series_info in chart_series_info:
            if 'NoLeak' not in series_info['sheet_name']:
                series_config = {
                    'name': series_info['sheet_name'],
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 11, series_info['data_points'], 11],     # SNR_Distance_dB column (column L)
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
                # Create SNR reference line for NoLeak measurements (should be around 0 dB for distance-specific)
                series_config = {
                    'name': series_info['sheet_name'] + ' (Distance Ref)',
                    'categories': [series_info['sheet_name'], 1, 2, series_info['data_points'], 2],  # Frequency column (column C)
                    'values': [series_info['sheet_name'], 1, 11, series_info['data_points'], 11],     # SNR_Distance_dB column (column L)
                    'line': {'width': 1}
                }
                # Apply grey color
                grey_color = grey_colors[grey_index % len(grey_colors)]
                series_config['line']['color'] = grey_color
                grey_index += 1
                chartSNRDist.add_series(series_config)
        
        # Configure distance-specific SNR chart
        chartSNRDist.set_title({'name': 'Distance-Specific Signal-to-Noise Ratio vs Frequency', 'name_font': {'size': 12}})
        chartSNRDist.set_x_axis({
            'name': 'Frequency (Hz)',
            'log_base': 10,
            'label_position': 'low',
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartSNRDist.set_y_axis({
            'name': 'SNR Distance-Specific (dB)',
            'label_position': 'low',
            'name_layout': {'x': 0.02, 'y': 0.5},
            'name_font': {'size': 12},
            'num_font': {'size': 12}
        })
        chartSNRDist.set_size({'width': 1400, 'height': 700})
        chartSNRDist.set_plotarea({'layout': {'x': 0.15, 'y': 0.15, 'width': 0.75, 'height': 0.70}})
        chartSNRDist.set_legend({'font': {'size': 12}})
        
        # Insert distance-specific SNR chart into worksheet
        snrDistanceWorksheet.insert_chart('A1', chartSNRDist)
    
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

        # Create Distance-Specific Leak Detection Results worksheet
        if leak_detection_distance_specific and isinstance(leak_detection_distance_specific, dict):
            distance_detection_sheet_name = 'Distance_Specific_Detection'
            counter = 1
            while distance_detection_sheet_name in used_worksheet_names:
                distance_detection_sheet_name = f'Distance_Specific_Detection_{counter}'
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
            
            # Add distance-specific frequency band analysis details
            row += 2
            distanceDetectionWorksheet.write(row, 0, 'Distance-Specific Frequency Band Analysis:')
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
            
            # Add distance-specific summary statistics
            row += 2
            distanceDetectionWorksheet.write(row, 0, 'Distance-Specific Detection Summary:')
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
    
    # Process and analyze this folder's data
    process_folder_analysis(subfolder_path, subfolder, folder_data)

print(f"\n" + "="*50)
print(f"🎵 ANALYSIS COMPLETE")
print(f"="*50)
print(f"- Total subfolders scanned: {len(subfolders)}")
print(f"- Total WAV files with distance pattern processed: {total_wav_files}")
print(f"- Total segments extracted: {total_segments}")
distance_pattern = r'\d+m'
print(f"- Analysis files created: {len([f for f in subfolders if len([x for x in os.listdir(os.path.join(path, f)) if x.lower().endswith('.wav') and re.search(distance_pattern, x)]) > 0])}")
print(f"\nNote: Only WAV files containing distance pattern (number + 'm') were processed.")
print(f"Each folder with valid files now contains its own analysis summary file.")

print('OK')
