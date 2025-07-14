#!/usr/bin/env python3
"""
Enhanced Acoustics FFT Analyzer
Processes acoustic data files and performs frequency domain analysis with visualization.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration parameters for acoustic analysis"""
    n_samples: int = 65536  # 2^16 samples
    n_avg: int = 4  # Number of averages
    downsample: int = 2  # Downsampling factor
    skip_rows: int = 1  # Rows to skip in CSV files
    output_format: str = 'both'  # 'excel', 'csv', or 'both'
    create_plots: bool = True
    plot_format: str = 'png'
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AnalysisConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()

class AcousticsAnalyzer:
    """Main class for acoustic data analysis"""
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.results = []
        
    def load_data_files(self, data_path: str) -> List[Tuple[str, pd.DataFrame]]:
        """Load all data files from specified directory"""
        data_path = Path(data_path)
        
        if not data_path.exists():
            logger.error(f"Data path {data_path} does not exist")
            return []
            
        files = []
        file_extensions = ['.csv', '.txt', '.dat']
        
        for ext in file_extensions:
            files.extend(data_path.glob(f'*{ext}'))
            
        if not files:
            logger.warning(f"No data files found in {data_path}")
            return []
            
        loaded_data = []
        
        for file_path in files:
            try:
                # Clean filename for later use
                filename = file_path.stem.replace(' ', '_')
                
                # Try different delimiters
                for delimiter in ['\s+', ',', '\t', ';']:
                    try:
                        df = pd.read_csv(file_path, header=None, skiprows=self.config.skip_rows, 
                                       delimiter=delimiter, engine='python')
                        if df.shape[1] >= 2:  # Need at least time and data columns
                            loaded_data.append((filename, df))
                            logger.info(f"Loaded {file_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
                            break
                    except Exception:
                        continue
                else:
                    logger.warning(f"Could not load {file_path.name}")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
                
        return loaded_data
    
    def calculate_sample_rate(self, time_array: np.ndarray) -> float:
        """Calculate sample rate from time array"""
        if len(time_array) < 2:
            raise ValueError("Need at least 2 time points to calculate sample rate")
            
        time_diffs = np.diff(time_array)
        avg_time_step = np.mean(time_diffs)
        
        # Check for consistent sampling
        time_std = np.std(time_diffs)
        if time_std / avg_time_step > 0.01:  # 1% tolerance
            logger.warning(f"Inconsistent sampling detected (std/mean = {time_std/avg_time_step:.3f})")
            
        return 1.0 / avg_time_step
    
    def perform_fft_analysis(self, data: pd.DataFrame, filename: str) -> Optional[dict]:
        """Perform FFT analysis on a single dataset"""
        try:
            # Downsample if requested
            if self.config.downsample > 1:
                data = data.iloc[::self.config.downsample]
                
            data_length = len(data)
            
            if self.config.n_avg > data_length // self.config.n_samples:
                logger.warning(f"Not enough data for {self.config.n_avg} averages in {filename}")
                return None
                
            # Extract time array
            time_array = data.iloc[0:self.config.n_samples, 0].to_numpy()
            
            # Calculate sampling parameters
            sample_rate = self.calculate_sample_rate(time_array)
            time_step = 1.0 / sample_rate
            
            # Generate frequency array
            freq_array = np.fft.fftfreq(self.config.n_samples, time_step)
            
            # Process multiple segments for averaging
            fft_results = []
            data_segments = []
            
            for i in range(self.config.n_avg):
                start_idx = i * self.config.n_samples
                end_idx = (i + 1) * self.config.n_samples
                
                data_segment = data.iloc[start_idx:end_idx, 1].to_numpy()
                data_segments.append(data_segment)
                
                # Perform FFT
                fft_result = np.fft.fft(data_segment)
                fft_results.append(fft_result)
            
            # Calculate statistics
            fft_array = np.array(fft_results)
            fft_avg = np.mean(fft_array, axis=0)
            fft_std = np.std(fft_array, axis=0)
            fft_min = np.min(fft_array, axis=0)
            fft_max = np.max(fft_array, axis=0)
            
            # Calculate magnitudes (normalized)
            fft_abs_avg = np.abs(fft_avg) / self.config.n_samples * 2
            fft_abs_std = np.abs(fft_std) / self.config.n_samples * 2
            fft_abs_min = np.abs(fft_min) / self.config.n_samples * 2
            fft_abs_max = np.abs(fft_max) / self.config.n_samples * 2
            
            # Calculate power spectral density
            psd = np.abs(fft_avg) ** 2 / (sample_rate * self.config.n_samples)
            
            return {
                'filename': filename,
                'sample_rate': sample_rate,
                'time_array': time_array,
                'freq_array': freq_array,
                'data_segments': data_segments,
                'fft_avg': fft_avg,
                'fft_std': fft_std,
                'fft_min': fft_min,
                'fft_max': fft_max,
                'magnitude_avg': fft_abs_avg,
                'magnitude_std': fft_abs_std,
                'magnitude_min': fft_abs_min,
                'magnitude_max': fft_abs_max,
                'psd': psd,
                'n_averages': self.config.n_avg
            }
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return None
    
    def save_results(self, result: dict, output_dir: str = "output"):
        """Save analysis results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = result['filename']
        n_half = self.config.n_samples // 2  # Only positive frequencies
        
        # Prepare data for saving
        save_data = {
            'Time_s': result['time_array'][:n_half],
            'Data_Average': np.mean(result['data_segments'], axis=0)[:n_half],
            'Frequency_Hz': result['freq_array'][:n_half],
            'FFT_Real_Avg': result['fft_avg'][:n_half].real,
            'FFT_Imag_Avg': result['fft_avg'][:n_half].imag,
            'Magnitude_Avg': result['magnitude_avg'][:n_half],
            'Magnitude_Std': result['magnitude_std'][:n_half],
            'PSD': result['psd'][:n_half]
        }
        
        df_output = pd.DataFrame(save_data)
        
        # Save as CSV
        if self.config.output_format in ['csv', 'both']:
            csv_path = output_path / f"{filename}_analysis.csv"
            df_output.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV: {csv_path}")
        
        # Save as Excel (if xlsxwriter is available)
        if self.config.output_format in ['excel', 'both']:
            try:
                import xlsxwriter
                excel_path = output_path / f"{filename}_analysis.xlsx"
                
                with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                    df_output.to_excel(writer, sheet_name='FFT_Analysis', index=False)
                    
                    # Add metadata sheet
                    metadata = pd.DataFrame({
                        'Parameter': ['Sample_Rate_Hz', 'N_Samples', 'N_Averages', 'Downsample_Factor'],
                        'Value': [result['sample_rate'], self.config.n_samples, 
                                result['n_averages'], self.config.downsample]
                    })
                    metadata.to_excel(writer, sheet_name='Metadata', index=False)
                    
                logger.info(f"Saved Excel: {excel_path}")
            except ImportError:
                logger.warning("xlsxwriter not available, skipping Excel output")
    
    def create_plots(self, result: dict, output_dir: str = "output"):
        """Create visualization plots"""
        if not self.config.create_plots:
            return
            
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = result['filename']
        n_half = self.config.n_samples // 2
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Acoustic Analysis: {filename}', fontsize=16)
        
        # Time domain plot
        time_data_avg = np.mean(result['data_segments'], axis=0)
        axes[0, 0].plot(result['time_array'], time_data_avg)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Time Domain Signal')
        axes[0, 0].grid(True)
        
        # Frequency domain magnitude
        freq_positive = result['freq_array'][:n_half]
        mag_positive = result['magnitude_avg'][:n_half]
        axes[0, 1].semilogy(freq_positive, mag_positive)
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].set_title('Frequency Domain (Log Scale)')
        axes[0, 1].grid(True)
        
        # Power Spectral Density
        psd_positive = result['psd'][:n_half]
        axes[1, 0].semilogy(freq_positive, psd_positive)
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('PSD (V²/Hz)')
        axes[1, 0].set_title('Power Spectral Density')
        axes[1, 0].grid(True)
        
        # Phase plot
        phase = np.angle(result['fft_avg'][:n_half])
        axes[1, 1].plot(freq_positive, phase)
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Phase (rad)')
        axes[1, 1].set_title('Phase Response')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / f"{filename}_analysis.{self.config.plot_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {plot_path}")
    
    def analyze_directory(self, data_path: str, output_dir: str = "output") -> List[dict]:
        """Analyze all files in a directory"""
        logger.info(f"Starting analysis of directory: {data_path}")
        
        # Load data files
        data_files = self.load_data_files(data_path)
        
        if not data_files:
            logger.error("No valid data files found")
            return []
        
        results = []
        
        for filename, dataframe in data_files:
            logger.info(f"Processing {filename}...")
            
            result = self.perform_fft_analysis(dataframe, filename)
            
            if result:
                self.save_results(result, output_dir)
                self.create_plots(result, output_dir)
                results.append(result)
                logger.info(f"Successfully processed {filename}")
            else:
                logger.error(f"Failed to process {filename}")
        
        logger.info(f"Analysis complete. Processed {len(results)} files.")
        return results

def create_sample_data(output_dir: str = "sample_data", n_files: int = 2):
    """Create sample acoustic data files for testing"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for i in range(n_files):
        # Generate synthetic acoustic signal
        duration = 1.0  # seconds
        sample_rate = 44100  # Hz
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a signal with multiple frequency components
        frequencies = [440, 880, 1320]  # A4, A5, E6 notes
        amplitudes = [1.0, 0.5, 0.3]
        
        signal = np.zeros_like(t)
        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * t)
        
        # Add some noise
        noise_level = 0.1
        noise = noise_level * np.random.randn(len(t))
        signal += noise
        
        # Save to CSV
        data = np.column_stack([t, signal])
        filename = output_path / f"sample_acoustic_data_{i+1}.csv"
        
        # Add header
        with open(filename, 'w') as f:
            f.write("Time_s,Amplitude\n")
            np.savetxt(f, data, delimiter=',', fmt='%.6f')
        
        logger.info(f"Created sample file: {filename}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Acoustics FFT Analyzer')
    parser.add_argument('--data-path', type=str, default='sample_data',
                       help='Path to directory containing acoustic data files')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample data files for testing')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
        logger.info("Sample data created. Run again without --create-sample to analyze.")
        return
    
    # Load configuration
    config = AnalysisConfig.from_file(args.config)
    
    # Create analyzer and run analysis
    analyzer = AcousticsAnalyzer(config)
    results = analyzer.analyze_directory(args.data_path, args.output_dir)
    
    if results:
        logger.info(f"Analysis completed successfully for {len(results)} files")
    else:
        logger.error("No files were successfully analyzed")

if __name__ == "__main__":
    main()