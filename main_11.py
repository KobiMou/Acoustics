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
            result = np.full_like(numerator, default, dtype=float)
            valid_mask = (denominator != 0) & np.isfinite(denominator) & np.isfinite(numerator)
            if np.any(valid_mask):
                result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
            return result
        else:
            # Handle scalar
            if denominator != 0 and np.isfinite(denominator) and np.isfinite(numerator):
                return numerator / denominator
            else:
                return default
    except:
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

def sanitize_worksheet_name(name, used_names=None):
    """
    Sanitize worksheet name to comply with Excel's 31-character limit
    and ensure uniqueness within the workbook.
    
    Args:
        name: Original worksheet name
        used_names: Set of already used names to avoid duplicates
    
    Returns:
        Sanitized name that is <= 31 characters and unique
    """
    if used_names is None:
        used_names = set()
    
    # Excel worksheet name restrictions:
    # - Max 31 characters
    # - Cannot contain: [ ] : * ? / \
    # - Cannot start or end with apostrophe
    
    # Remove invalid characters
    invalid_chars = ['[', ']', ':', '*', '?', '/', '\\', "'"]
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Remove leading/trailing apostrophes
    sanitized = sanitized.strip("'")
    
    # Truncate to 31 characters
    if len(sanitized) <= 31:
        base_name = sanitized
        suffix = ""
    else:
        # Reserve space for potential suffix (_1, _2, etc.)
        base_name = sanitized[:27]  # Leave 4 chars for suffix like "_123"
        suffix = ""
        print(f"  Warning: Worksheet name truncated: '{name}' -> '{base_name}...'")
        print(f"           Original length: {len(name)}, Truncated length: {len(base_name)}")
    
    # Ensure uniqueness
    final_name = base_name + suffix
    counter = 1
    
    while final_name in used_names:
        # Create suffix with counter
        suffix = f"_{counter}"
        # Adjust base name length to accommodate suffix
        max_base_length = 31 - len(suffix)
        adjusted_base = base_name[:max_base_length]
        final_name = adjusted_base + suffix
        counter += 1
        
        # Safety check to prevent infinite loop
        if counter > 999:
            final_name = f"Sheet_{counter}"
            break
    
    # Log if we had to add a suffix for uniqueness
    if suffix:
        print(f"  Info: Added suffix for uniqueness: '{base_name}' -> '{final_name}'")
    
    used_names.add(final_name)
    return final_name

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
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    
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
                
                #fftFreq = scipy.fftpack.fftfreq(N, st)
                fft.append(np.fft.fft(windowed_data))
                #fft = scipy.fftpack.fft(Array[1])
                fftAbs = np.abs(fft)/N*2*2 # Normalize result for correct amplitude (×2 for Hanning window compensation)
            
            fft_AVG = np.mean(fft, axis=0)  
            fftAbs_AVG = np.mean(fftAbs, axis=0)    
            fft_MIN = np.min(fft, axis=0)  
            fftAbs_MIN = np.min(fftAbs, axis=0)
            
            # Calculate PSD (Power Spectral Density) - already using windowed FFT results
            # Hanning window correction factor for PSD (compensate for power loss)
            window_power = np.mean(hanning_window**2)
            psd_AVG = (np.abs(fft_AVG)**2) / (fs * N * window_power)
            
            # Calculate SNR
            # Simple SNR calculation: signal power / noise floor
            signal_power = np.abs(fft_AVG)**2
            noise_floor = np.median(signal_power)  # Use median as noise floor estimate
            snr_linear = safe_divide(signal_power, noise_floor, default=1e-10)
            snr_dB = 10 * safe_log10(snr_linear, default=-100)
            
            # Distance-specific SNR calculation
            distance_match = re.search(r'(\d+)m', ListFileNames[iLoop])
            distance = int(distance_match.group(1)) if distance_match else 1
            distance_factor = 1 / (distance**2) if distance > 0 else 1  # Inverse square law
            snr_distance_linear = snr_linear * distance_factor
            snr_distance_dB = 10 * safe_log10(snr_distance_linear, default=-100)
            
            # Sanitize worksheet name to comply with Excel's 31-character limit
            sanitized_sheet_name = sanitize_worksheet_name(ListFileNames[iLoop], used_worksheet_names)
            
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
                'psd': psd_AVG[:N//2],
                'fft_abs': fftAbs_AVG[:N//2],
                'snr_db': snr_dB[:N//2]
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
    
    # Leak detection functions (keeping the same as before)
    def detect_leak_statistical(leak_psd, noleak_baseline, confidence_factor=3):
        """
        Detect leak using statistical analysis
        """
        try:
            # Calculate statistics for baseline
            baseline_mean = np.mean(noleak_baseline)
            baseline_std = np.std(noleak_baseline)
            
            # Calculate threshold
            threshold = baseline_mean + confidence_factor * baseline_std
            
            # Calculate leak probability
            leak_indicators = leak_psd > threshold
            leak_probability = np.mean(leak_indicators)
            
            return {
                'method': 'statistical',
                'threshold': threshold,
                'leak_probability': leak_probability,
                'confidence_factor': confidence_factor,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std
            }
        except Exception as e:
            return {'method': 'statistical', 'error': str(e)}
    
    def detect_leak_frequency_bands(freq, leak_psd, noleak_baseline):
        """
        Detect leak using frequency band analysis
        """
        try:
            # Define frequency bands for leak detection (in Hz)
            leak_bands = [
                (50, 200),    # Low frequency structural vibrations
                (200, 500),   # Pipe resonances
                (500, 1000),  # Leak-specific frequencies
                (1000, 2000), # High frequency leaks
                (2000, 5000), # Very high frequency leaks
                (5000, 10000), # Ultrasonic range
                (10000, 15000), # Extended ultrasonic
                (15000, 20000)  # High ultrasonic
            ]
            
            band_results = {}
            
            for band_low, band_high in leak_bands:
                # Find frequency indices for this band
                band_indices = np.where((freq >= band_low) & (freq < band_high))[0]
                
                if len(band_indices) > 0:
                    # Calculate power in this band
                    leak_power = np.mean(leak_psd[band_indices])
                    baseline_power = np.mean(noleak_baseline[band_indices])
                    
                    # Calculate ratio
                    power_ratio = safe_divide(leak_power, baseline_power, default=0)
                    
                    band_results[f'{band_low}-{band_high}Hz'] = {
                        'leak_power': leak_power,
                        'baseline_power': baseline_power,
                        'power_ratio': power_ratio,
                        'ratio_db': 10 * safe_log10(power_ratio, default=-100)
                    }
            
            return {
                'method': 'frequency_bands',
                'bands': band_results
            }
        except Exception as e:
            return {'method': 'frequency_bands', 'error': str(e)}
    
    def detect_leak_power_ratio(leak_psd, noleak_baseline, threshold_dB=8):
        """
        Detect leak using overall power ratio
        """
        try:
            leak_power = np.mean(leak_psd)
            baseline_power = np.mean(noleak_baseline)
            
            power_ratio = safe_divide(leak_power, baseline_power, default=0)
            ratio_dB = 10 * safe_log10(power_ratio, default=-100)
            
            leak_detected = ratio_dB > threshold_dB
            
            return {
                'method': 'power_ratio',
                'leak_power': leak_power,
                'baseline_power': baseline_power,
                'power_ratio': power_ratio,
                'ratio_dB': ratio_dB,
                'threshold_dB': threshold_dB,
                'leak_detected': leak_detected
            }
        except Exception as e:
            return {'method': 'power_ratio', 'error': str(e)}
    
    def calculate_leak_detection_score(freq, leak_psd, noleak_baseline):
        """
        Calculate comprehensive leak detection score
        """
        try:
            # Statistical detection
            stat_result = detect_leak_statistical(leak_psd, noleak_baseline)
            
            # Power ratio detection
            power_result = detect_leak_power_ratio(leak_psd, noleak_baseline)
            
            # Frequency bands detection
            bands_result = detect_leak_frequency_bands(freq, leak_psd, noleak_baseline)
            
            # Calculate overall score
            score_components = []
            
            # Statistical score
            if 'leak_probability' in stat_result:
                score_components.append(stat_result['leak_probability'])
            
            # Power ratio score
            if 'leak_detected' in power_result:
                score_components.append(1.0 if power_result['leak_detected'] else 0.0)
            
            # Frequency bands score
            if 'bands' in bands_result:
                high_ratio_bands = sum(1 for band in bands_result['bands'].values() 
                                     if band.get('ratio_db', -np.inf) > 5)
                bands_score = high_ratio_bands / len(bands_result['bands'])
                score_components.append(bands_score)
            
            overall_score = np.mean(score_components) if score_components else 0.0
            
            return {
                'overall_score': overall_score,
                'statistical': stat_result,
                'power_ratio': power_result,
                'frequency_bands': bands_result,
                'components_used': len(score_components)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_leak_detection(analysis_data):
        """
        Analyze leak detection for all files
        """
        results = {}
        
        # Group files by base name (without leak/noleak suffix)
        file_groups = {}
        for data in analysis_data:
            base_name = data['filename'].replace('_leak', '').replace('_noleak', '')
            if base_name not in file_groups:
                file_groups[base_name] = {}
            
            if '_leak' in data['filename']:
                file_groups[base_name]['leak'] = data
            elif '_noleak' in data['filename']:
                file_groups[base_name]['noleak'] = data
        
        # Analyze each pair
        for base_name, group in file_groups.items():
            if 'leak' in group and 'noleak' in group:
                leak_data = group['leak']
                noleak_data = group['noleak']
                
                # Perform leak detection analysis
                detection_result = calculate_leak_detection_score(
                    leak_data['frequency'], 
                    leak_data['psd'], 
                    noleak_data['psd']
                )
                
                detection_result['leak_filename'] = leak_data['filename']
                detection_result['noleak_filename'] = noleak_data['filename']
                detection_result['baseline_available'] = True
                
                results[leak_data['filename']] = detection_result
        
        return results
    
    # Calculate and update SNR values in worksheets
    if analysis_data:
        # Global baseline for SNR calculation
        all_psd = np.concatenate([data['psd'] for data in analysis_data])
        global_noise_floor = np.median(all_psd)
        
        # Create global noise floor baseline
        baseline_psd = np.full_like(analysis_data[0]['psd'], global_noise_floor)
        
        # Perform leak detection analysis
        leak_detection_results = analyze_leak_detection(analysis_data)
        
        # Update the worksheet with SNR values
        for i, data in enumerate(analysis_data):
            filename = data['filename']
            
            # Find the corresponding sanitized sheet name
            sanitized_sheet_name = None
            for series_info in chart_series_info:
                if series_info['filename'] == filename:
                    sanitized_sheet_name = series_info['sheet_name']
                    break
            
            if sanitized_sheet_name:
                # Find and update the corresponding worksheet
                for ws in summaryWorkbook.worksheets():
                    if ws.get_name() == sanitized_sheet_name:
                        # Update SNR calculations with global baseline
                        snr_ratio = safe_divide(data['psd'], global_noise_floor, default=1e-10)
                        updated_snr = 10 * safe_log10(snr_ratio, default=-100)
                        for row in range(len(updated_snr)):
                            ws.write(row+1, 10, sanitize_excel_value(updated_snr[row]))
                        break
    
    def analyze_leak_detection_distance_specific(analysis_data):
        """
        Analyze leak detection with distance-specific considerations
        """
        results = {}
        
        # Group by distance
        distance_groups = {}
        for data in analysis_data:
            distance_match = re.search(r'(\d+)m', data['filename'])
            distance = int(distance_match.group(1)) if distance_match else 0
            
            if distance not in distance_groups:
                distance_groups[distance] = []
            distance_groups[distance].append(data)
        
        # Analyze each distance group
        for distance, group_data in distance_groups.items():
            # Create distance-specific baseline
            noleak_files = [data for data in group_data if '_noleak' in data['filename']]
            if noleak_files:
                combined_baseline = np.mean([data['psd'] for data in noleak_files], axis=0)
                
                # Analyze leak files against this baseline
                leak_files = [data for data in group_data if '_leak' in data['filename']]
                for leak_data in leak_files:
                    detection_result = calculate_leak_detection_score(
                        leak_data['frequency'], 
                        leak_data['psd'], 
                        combined_baseline
                    )
                    
                    detection_result['distance'] = distance
                    detection_result['baseline_type'] = 'distance_specific'
                    
                    results[leak_data['filename']] = detection_result
        
        return results
    
    # Perform distance-specific leak detection analysis
    leak_detection_distance_specific = analyze_leak_detection_distance_specific(analysis_data)
    
    # Create plot worksheet
    if chart_series_info:
        plot_sheet_name = sanitize_worksheet_name('Plot', used_worksheet_names)
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
            if 'noleak' not in series_info['filename'].lower():
                series_config = {
                    'name': series_info['filename'],
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
            if 'noleak' in series_info['filename'].lower():
                series_config = {
                    'name': series_info['filename'],
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
            'name_font': {'size': 10},
            'num_font': {'size': 9},
            'min': 1,
            'max': 10000
        })
        chart.set_y_axis({
            'name': 'FFT Minimum Absolute Values',
            'name_font': {'size': 10},
            'num_font': {'size': 9},
            'min': 0
        })
        chart.set_size({'width': 1400, 'height': 700})
        chart.set_plotarea({'layout': {'x': 0.15, 'y': 0.15, 'width': 0.75, 'height': 0.70}})
        chart.set_legend({'font': {'size': 12}})
        
        # Insert chart into worksheet
        plotWorksheet.insert_chart('A1', chart)
        
        # Create second plot worksheet with logarithmic Y-axis
        plot_log_sheet_name = sanitize_worksheet_name('Plot_Log_Scale', used_worksheet_names)
        plotLogWorksheet = summaryWorkbook.add_worksheet(plot_log_sheet_name)
        
        # Create scatter chart with straight lines (same as first plot)
        chartLog = summaryWorkbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
        
        # Repeat the same series addition for log chart
        color_index = 0
        for series_info in chart_series_info:
            if 'noleak' not in series_info['filename'].lower():
                series_config = {
                    'name': series_info['filename'],
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
            if 'noleak' in series_info['filename'].lower():
                series_config = {
                    'name': series_info['filename'],
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
            'name_font': {'size': 10},
            'num_font': {'size': 9},
            'min': 1,
            'max': 10000
        })
        chartLog.set_y_axis({
            'name': 'FFT Minimum Absolute Values',
            'log_base': 10,
            'name_font': {'size': 10},
            'num_font': {'size': 9},
            'min': 0.0001
        })
        chartLog.set_size({'width': 1400, 'height': 700})
        chartLog.set_plotarea({'layout': {'x': 0.15, 'y': 0.15, 'width': 0.75, 'height': 0.70}})
        chartLog.set_legend({'font': {'size': 12}})
        
        # Insert chart into worksheet
        plotLogWorksheet.insert_chart('A1', chartLog)
    
    # Add leak detection results worksheet
    if leak_detection_distance_specific:
        leak_detection_sheet_name = sanitize_worksheet_name('Leak_Detection_Results', used_worksheet_names)
        leakDetectionWorksheet = summaryWorkbook.add_worksheet(leak_detection_sheet_name)
        
        # Write headers
        headers = ['Filename', 'Distance', 'Overall Score', 'Statistical Prob', 'Power Ratio (dB)', 'Leak Detected', 'Baseline Type']
        for col, header in enumerate(headers):
            leakDetectionWorksheet.write(0, col, header)
        
        row = 1
        for filename, result in leak_detection_distance_specific.items():
            leakDetectionWorksheet.write(row, 0, filename)
            leakDetectionWorksheet.write(row, 1, result.get('distance', 'N/A'))
            
            # Overall score
            overall_score = result.get('overall_score', 'N/A')
            if overall_score != 'N/A':
                overall_score = sanitize_excel_value(overall_score)
            leakDetectionWorksheet.write(row, 2, overall_score)
            
            # Statistical probability
            stat_prob = 'N/A'
            if 'statistical' in result and 'leak_probability' in result['statistical']:
                stat_prob = sanitize_excel_value(result['statistical']['leak_probability'])
            leakDetectionWorksheet.write(row, 3, stat_prob)
            
            # Power ratio
            power_ratio = 'N/A'
            if 'power_ratio' in result and 'ratio_dB' in result['power_ratio']:
                power_ratio = sanitize_excel_value(result['power_ratio']['ratio_dB'])
            leakDetectionWorksheet.write(row, 4, power_ratio)
            
            # Leak detected
            leak_detected = 'N/A'
            if 'power_ratio' in result and 'leak_detected' in result['power_ratio']:
                leak_detected = result['power_ratio']['leak_detected']
            leakDetectionWorksheet.write(row, 5, leak_detected)
            
            leakDetectionWorksheet.write(row, 6, result.get('baseline_type', 'N/A'))
            row += 1
        
        # Add summary statistics
        row += 2
        leakDetectionWorksheet.write(row, 0, 'Summary Statistics')
        row += 1
        
        # Calculate summary by distance
        distance_summary = {}
        for filename, result in leak_detection_distance_specific.items():
            distance = result.get('distance', 'Unknown')
            if distance not in distance_summary:
                distance_summary[distance] = {'total': 0, 'high': 0, 'medium': 0, 'low': 0, 'no_baseline': 0}
            
            distance_summary[distance]['total'] += 1
            
            score = result.get('overall_score', 0)
            if score > 0.7:
                distance_summary[distance]['high'] += 1
            elif score > 0.4:
                distance_summary[distance]['medium'] += 1
            elif score > 0.1:
                distance_summary[distance]['low'] += 1
            else:
                distance_summary[distance]['no_baseline'] += 1
        
        # Write summary by distance
        summary_headers = ['Distance (m)', 'Total Measurements', 'High Prob', 'Medium Prob', 'Low Prob', 'No Baseline']
        for col, header in enumerate(summary_headers):
            leakDetectionWorksheet.write(row, col, header)
        row += 1
        
        for distance in sorted(distance_summary.keys()):
            summary = distance_summary[distance]
            leakDetectionWorksheet.write(row, 0, distance)
            leakDetectionWorksheet.write(row, 1, summary['total'])
            leakDetectionWorksheet.write(row, 2, summary['high'])
            leakDetectionWorksheet.write(row, 3, summary['medium'])
            leakDetectionWorksheet.write(row, 4, summary['low'])
            leakDetectionWorksheet.write(row, 5, summary['no_baseline'])
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
    wav_files_in_folder = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.wav')]
    total_wav_files += len(wav_files_in_folder)
    
    print(f"\nProcessing folder: {subfolder} ({len(wav_files_in_folder)} WAV files)")
    
    # Process WAV files in this subfolder
    folder_data = process_wav_files_in_folder(subfolder_path)
    total_segments += len(folder_data)
    
    # Process and analyze this folder's data
    process_folder_analysis(subfolder_path, subfolder, folder_data)

print(f"\n" + "="*50)
print(f"🎵 ANALYSIS COMPLETE")
print(f"="*50)
print(f"- Total subfolders processed: {len(subfolders)}")
print(f"- Total WAV files processed: {total_wav_files}")
print(f"- Total segments extracted: {total_segments}")
print(f"- Analysis files created: {len(subfolders)}")
print(f"\nEach folder now contains its own analysis summary file.")

print('OK')
