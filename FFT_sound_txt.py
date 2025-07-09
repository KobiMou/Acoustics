import os
import numpy as np
import pandas as pd
import xlsxwriter
import scipy.fftpack
import matplotlib.pyplot as plt
import re

path = r'D:\OneDrive - Arad Technologies Ltd\ARAD_Projects\ALD\tests\test_07_07_2025_main\Edit_data'

files = os.listdir(path)

ListFileNames = []
ListDataFrames = []

n_samples = 65536   # 2^n  -> 65536
n_AVG = 4
downsample = 2  # take every "downsample" row, reduces fs -> lower Fnyq/2 but longer time -> better freq res

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

# Store plot data for all files
plot_data = []

iLoop = 0

for DataFrame in ListDataFrames:
    DataFrameSize = len(DataFrame)
    
    # Downsample
    if downsample > 1:
        DataFrame = DataFrame.iloc[::downsample]
    
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
        
        for j in range(n_AVG):
            data_Array = DataFrame[1][j*n_samples:(j+1)*n_samples].to_numpy()
            data_Array = np.transpose(data_Array)

            Array_list.append(data_Array)

            
            #fftFreq = scipy.fftpack.fftfreq(N, st)
            fft.append(np.fft.fft(data_Array))
            #fft = scipy.fftpack.fft(Array[1])
            fftAbs = np.abs(fft)/N*2 # Normalize result for correct amplitude
        
        fft_AVG = np.mean(fft, axis=0)  
        fftAbs_AVG = np.mean(fftAbs, axis=0)    
        fft_MIN = np.min(fft, axis=0)  
        fftAbs_MIN = np.min(fftAbs, axis=0)      

        # Store plot data for this file
        plot_data.append({
            'filename': ListFileNames[iLoop],
            'fftFreq': fftFreq[:N//2],
            'fftAbs_MIN': fftAbs_MIN[:N//2]
        })

        # Create individual workbook for each file
        workbookName = ListFileNames[iLoop] + '_Output' + '.xlsx'
        workbook = xlsxwriter.Workbook(workbookName)
        worksheet = workbook.add_worksheet()
        worksheet.name = 'Data'

        # Create worksheet in summary workbook for this file
        summaryWorksheet = summaryWorkbook.add_worksheet(ListFileNames[iLoop])
        
        # Write headers for summary worksheet
        headers = ['Time', 'Data', 'Frequency', 'FFT_AVG_Real', 'FFT_AVG_Imag', 'FFT_AVG_Abs', 'FFT_MIN_Real', 'FFT_MIN_Imag', 'FFT_MIN_Abs']
        for col, header in enumerate(headers):
            summaryWorksheet.write(0, col, header)

        for i in range(N//2): # //2 for only positive side plotting 
            # Write to individual file
            worksheet.write(i,0,time_Array[i])
            worksheet.write(i,1,data_Array[i])
            worksheet.write(i,2,fftFreq[i])
            worksheet.write(i,3,fft_AVG[i].real)
            worksheet.write(i,4,fft_AVG[i].imag)
            worksheet.write(i,5,fftAbs_AVG[i])
            worksheet.write(i,6,fft_MIN[i].real)
            worksheet.write(i,7,fft_MIN[i].imag)
            worksheet.write(i,8,fftAbs_MIN[i])
            
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
        

        workbook.close()
        iLoop = iLoop + 1

# Create plot worksheet
if plot_data:
    plotWorksheet = summaryWorkbook.add_worksheet('Plot')
    
    # Write headers
    header_row = ['Frequency']
    for data in plot_data:
        header_row.append(f"{data['filename']}_FFT_Min_Abs")
    
    for col, header in enumerate(header_row):
        plotWorksheet.write(0, col, header)
    
    # Write frequency data (using first file's frequency array as reference)
    freq_data = plot_data[0]['fftFreq']
    for row, freq in enumerate(freq_data):
        plotWorksheet.write(row + 1, 0, freq)
    
    # Write FFT minimum absolute values for each file
    for col, data in enumerate(plot_data):
        for row, value in enumerate(data['fftAbs_MIN']):
            plotWorksheet.write(row + 1, col + 1, value)
    
    # Create chart
    chart = summaryWorkbook.add_chart({'type': 'line'})
    
    # Add series for each file
    for col, data in enumerate(plot_data):
        chart.add_series({
            'name': data['filename'],
            'categories': ['Plot', 1, 0, len(freq_data), 0],
            'values': ['Plot', 1, col + 1, len(freq_data), col + 1],
            'line': {'width': 2}
        })
    
    # Configure chart
    chart.set_title({'name': 'FFT Frequency vs Minimum Absolute Values'})
    chart.set_x_axis({'name': 'Frequency (Hz)'})
    chart.set_y_axis({'name': 'FFT Minimum Absolute Values'})
    chart.set_size({'width': 1200, 'height': 600})
    
    # Insert chart into worksheet
    plotWorksheet.insert_chart('A' + str(len(freq_data) + 5), chart)

# Close summary workbook
summaryWorkbook.close()

print('OK')
