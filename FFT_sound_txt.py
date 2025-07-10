import os
import numpy as np
import pandas as pd
import xlsxwriter
import scipy.fftpack
import matplotlib.pyplot as plt
import re

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

# Store chart series info
chart_series_info = []

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
        psd_MIN = np.min(psd, axis=0)      

        # Store chart series info for this file
        chart_series_info.append({
            'filename': ListFileNames[iLoop],
            'sheet_name': ListFileNames[iLoop],
            'data_points': N//2
        })

        # Create individual workbook for each file
        # workbookName = ListFileNames[iLoop] + '_Output' + '.xlsx'
        # workbook = xlsxwriter.Workbook(workbookName)
        # worksheet = workbook.add_worksheet()
        # worksheet.name = 'Data'

        # Create worksheet in summary workbook for this file
        summaryWorksheet = summaryWorkbook.add_worksheet(ListFileNames[iLoop])
        
        # Write headers for summary worksheet
        headers = ['Time', 'Data', 'Frequency', 'FFT_AVG_Real', 'FFT_AVG_Imag', 'FFT_AVG_Abs', 'FFT_MIN_Real', 'FFT_MIN_Imag', 'FFT_MIN_Abs', 'PSD_AVG', 'PSD_MIN']
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
            summaryWorksheet.write(i+1,10,psd_MIN[i])
        

        # workbook.close()
        iLoop = iLoop + 1

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
                'values': [series_info['sheet_name'], 1, 10, series_info['data_points'], 10],     # PSD_MIN column (column K)
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
                'values': [series_info['sheet_name'], 1, 10, series_info['data_points'], 10],     # PSD_MIN column (column K)
                'line': {'width': 2}
            }
            # Apply grey color in ascending order (darker to lighter)
            grey_color = grey_colors[grey_index % len(grey_colors)]
            series_config['line']['color'] = grey_color
            grey_index += 1
            chartPSD.add_series(series_config)
    
    # Configure PSD chart with logarithmic frequency axis
    chartPSD.set_title({'name': 'PSD Frequency vs Minimum Values', 'name_font': {'size': 12}})
    chartPSD.set_x_axis({
        'name': 'Frequency (Hz)',
        'log_base': 10,
        'label_position': 'low',
        'name_font': {'size': 12},
        'num_font': {'size': 12}
    })
    chartPSD.set_y_axis({
        'name': 'PSD Minimum Values',
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

# Close summary workbook
summaryWorkbook.close()

print('OK')
