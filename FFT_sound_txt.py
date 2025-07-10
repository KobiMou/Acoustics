import os
import numpy as np
import pandas as pd
import xlsxwriter
import scipy.fftpack
import matplotlib.pyplot as plt
import re
from scipy import stats

path = r'D:\OneDrive - Arad Technologies Ltd\ARAD_Projects\ALD\tests\test_07_07_2025_main\Edit_data\test_5ch_txt'

files = os.listdir(path)

ListFileNames = []
ListDataFrames = []

n_samples = 131072  # 2^n  -> 131072 (2^17)
n_AVG = 7

# Create list of tuples (filename, dataframe) for sorting
file_data_pairs = []

for f in files:
    fSplit = f.split('.')
    fName = fSplit[0]
    if ' ' in fName: fName = fName.replace(' ', '_')
    fullpath = os.path.join(path, f)
    skipRows = 1 # skip only header: 1, skip header and first data row: 2
    DataFrame = pd.read_csv(fullpath, header=None, skiprows=skipRows, delimiter='\s+')
    file_data_pairs.append((fName, DataFrame))

# Sort by numeric value before "m"
def extract_number_before_m(filename):
    match = re.search(r'(\d+)m', filename)
    return int(match.group(1)) if match else float('inf')

file_data_pairs.sort(key=lambda x: extract_number_before_m(x[0]))

# Extract sorted lists
for fName, DataFrame in file_data_pairs:
    ListFileNames.append(fName)
    ListDataFrames.append(DataFrame)

# Create summary workbook
summaryWorkbook = xlsxwriter.Workbook('summary.xlsx')

# Store chart series info and analysis data
chart_series_info = []
analysis_data = []  # Store frequency and PSD data for leak detection

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
        hanning_correction = 8/3  # Correction factor for Hanning window power
        psd = []
        for j in range(n_AVG):
            psd_single = (np.abs(fft[j])**2) / (N * fs) * hanning_correction
            psd.append(psd_single)
        
        psd_AVG = np.mean(psd, axis=0)      

        # Store chart series info for this file
        chart_series_info.append({
            'filename': ListFileNames[iLoop],
            'sheet_name': ListFileNames[iLoop],
            'data_points': N//2
        })
        
        # Store analysis data for leak detection
        analysis_data.append({
            'filename': ListFileNames[iLoop],
            'frequency': fftFreq[:N//2],
            'psd_avg': psd_AVG[:N//2],
            'fft_abs_min': fftAbs_MIN[:N//2],
            'is_noleak': 'NoLeak' in ListFileNames[iLoop]
        })

        # Create individual workbook for each file
        # workbookName = ListFileNames[iLoop] + '_Output' + '.xlsx'
        # workbook = xlsxwriter.Workbook(workbookName)
        # worksheet = workbook.add_worksheet()
        # worksheet.name = 'Data'

        # Create worksheet in summary workbook for this file
        summaryWorksheet = summaryWorkbook.add_worksheet(ListFileNames[iLoop])
        
        # Write headers for summary worksheet
        headers = ['Time', 'Data', 'Frequency', 'FFT_AVG_Real', 'FFT_AVG_Imag', 'FFT_AVG_Abs', 'FFT_MIN_Real', 'FFT_MIN_Imag', 'FFT_MIN_Abs', 'PSD_AVG']
        for col, header in enumerate(headers):
            summaryWorksheet.write(0, col, header)

        for i in range(N//2): # //2 for only positive side plotting 
            # Write to individual file
            # worksheet.write(i,0,time_Array[i])
            # worksheet.write(i,1,data_Array[i])
            # worksheet.write(i,2,fftFreq[i])
            # worksheet.write(i,3,fft_AVG[i].real)
            # worksheet.write(i,4,fft_AVG[i].imag)
            # worksheet.write(i,5,fftAbs_AVG[i])
            # worksheet.write(i,6,fft_MIN[i].real)
            # worksheet.write(i,7,fft_MIN[i].imag)
            # worksheet.write(i,8,fftAbs_MIN[i])
            
            # Write to summary file
            summaryWorksheet.write(i+1,0,time_Array[i])
            summaryWorksheet.write(i+1,1,data_Array[i])
            summaryWorksheet.write(i+1,2,fftFreq[i])
            summaryWorksheet.write(i+1,3,fft_AVG[i].real)
            summaryWorksheet.write(i+1,4,fft_AVG[i].imag)
            summaryWorksheet.write(i+1,5,fftAbs_AVG[i])
            summaryWorksheet.write(i+1,6,fft_MIN[i].real)
            summaryWorksheet.write(i+1,7,fft_MIN[i].imag)
            summaryWorksheet.write(i+1,8,fftAbs_MIN[i])
            summaryWorksheet.write(i+1,9,psd_AVG[i])
        

        # workbook.close()
        iLoop = iLoop + 1

# Leak Detection Functions
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
    plotWorksheet = summaryWorkbook.add_worksheet('Plot')
    
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
        if 'NoLeak' not in series_info['filename']:
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
        if 'NoLeak' in series_info['filename']:
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
    plotLogWorksheet = summaryWorkbook.add_worksheet('Plot_Log_Scale')
    
    # Create scatter chart with straight lines (same as first plot)
    chartLog = summaryWorkbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
    
    # First add all non-NoLeak series (colorful lines in background)
    color_index = 0
    for series_info in chart_series_info:
        if 'NoLeak' not in series_info['filename']:
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
        if 'NoLeak' in series_info['filename']:
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
    psdPlotWorksheet = summaryWorkbook.add_worksheet('PSD_Plot')
    
    # Create scatter chart with straight lines for PSD
    chartPSD = summaryWorkbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
    
    # First add all non-NoLeak series (colorful lines in background)
    color_index = 0
    for series_info in chart_series_info:
        if 'NoLeak' not in series_info['filename']:
            series_config = {
                'name': series_info['filename'],
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
        if 'NoLeak' in series_info['filename']:
            series_config = {
                'name': series_info['filename'],
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

    # Create Leak Detection Results worksheet
    if leak_detection_results and isinstance(leak_detection_results, dict):
        leakDetectionWorksheet = summaryWorkbook.add_worksheet('Leak_Detection')
        
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
        distanceDetectionWorksheet = summaryWorkbook.add_worksheet('Distance_Specific_Detection')
        
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

# Close summary workbook
summaryWorkbook.close()

print('OK')
