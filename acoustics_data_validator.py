#!/usr/bin/env python3
"""
Acoustics Data Validator
Comprehensive data quality assessment tool for acoustic measurements.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
from scipy import stats
from scipy.signal import find_peaks
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Container for data quality metrics"""
    filename: str
    file_size_mb: float
    n_samples: int
    sampling_rate: float
    duration_seconds: float
    signal_range: Tuple[float, float]
    signal_mean: float
    signal_std: float
    signal_snr_estimate: float
    missing_values: int
    outlier_count: int
    outlier_percentage: float
    time_gaps: List[Tuple[int, float]]
    sampling_consistency: float
    dc_offset: float
    frequency_content_score: float
    data_integrity_score: float
    recommendations: List[str]

class AcousticsDataValidator:
    """Comprehensive data validation for acoustic measurements"""
    
    def __init__(self, config_path: str = "acoustics_config.json"):
        self.config = self.load_config(config_path)
        self.validation_results = []
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Return default configuration"""
        return {
            "analysis_parameters": {
                "n_samples": 131072,
                "skip_rows": 1
            },
            "file_processing": {
                "supported_extensions": [".csv", ".txt", ".dat"],
                "delimiter_options": ["\\s+", ",", "\t", ";"]
            },
            "validation": {
                "min_samples": 1000,
                "max_outlier_percentage": 5.0,
                "min_sampling_rate": 1000,
                "max_sampling_rate": 100000,
                "max_time_gap_seconds": 0.001,
                "min_snr_db": 20,
                "max_dc_offset_percentage": 10
            }
        }
    
    def detect_outliers(self, data: np.ndarray, method: str = "iqr") -> Tuple[np.ndarray, int]:
        """Detect outliers in the data using specified method"""
        if method == "iqr":
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            outliers = z_scores > 3
        
        elif method == "modified_zscore":
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > 3.5
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers, np.sum(outliers)
    
    def assess_sampling_consistency(self, time_array: np.ndarray) -> Tuple[float, List[Tuple[int, float]]]:
        """Assess consistency of sampling intervals"""
        time_diffs = np.diff(time_array)
        
        # Calculate coefficient of variation
        cv = np.std(time_diffs) / np.mean(time_diffs) if np.mean(time_diffs) > 0 else float('inf')
        
        # Find significant time gaps
        median_diff = np.median(time_diffs)
        gap_threshold = median_diff * 3  # Gaps larger than 3x median
        
        gaps = []
        for i, diff in enumerate(time_diffs):
            if diff > gap_threshold:
                gaps.append((i, diff))
        
        return cv, gaps
    
    def estimate_snr(self, signal: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio"""
        # Simple SNR estimation using signal power vs noise floor
        signal_power = np.mean(signal**2)
        
        # Estimate noise floor from lower percentile
        noise_floor = np.percentile(np.abs(signal), 10)**2
        
        if noise_floor > 0:
            snr_linear = signal_power / noise_floor
            snr_db = 10 * np.log10(snr_linear)
        else:
            snr_db = float('inf')
        
        return snr_db
    
    def analyze_frequency_content(self, signal: np.ndarray, sampling_rate: float) -> float:
        """Analyze frequency content quality"""
        # Perform FFT
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal), 1/sampling_rate)
        magnitude = np.abs(fft_result)
        
        # Only analyze positive frequencies
        positive_freq_mask = frequencies > 0
        frequencies = frequencies[positive_freq_mask]
        magnitude = magnitude[positive_freq_mask]
        
        # Assess frequency content quality
        # 1. Check for spectral peaks
        peaks, _ = find_peaks(magnitude, height=np.max(magnitude) * 0.1)
        n_significant_peaks = len(peaks)
        
        # 2. Check frequency distribution
        # Good acoustic signals should have content across multiple frequency bands
        freq_bands = [(0, 100), (100, 1000), (1000, 5000), (5000, 20000)]
        bands_with_content = 0
        
        for f_low, f_high in freq_bands:
            band_mask = (frequencies >= f_low) & (frequencies <= f_high)
            if np.any(band_mask):
                band_energy = np.sum(magnitude[band_mask]**2)
                total_energy = np.sum(magnitude**2)
                if band_energy / total_energy > 0.01:  # At least 1% of energy
                    bands_with_content += 1
        
        # 3. Calculate spectral flatness (measure of how noise-like the signal is)
        # Closer to 1 means more noise-like, closer to 0 means more tonal
        geometric_mean = stats.gmean(magnitude + 1e-10)  # Add small value to avoid log(0)
        arithmetic_mean = np.mean(magnitude)
        spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        
        # Combine metrics into a quality score (0-100)
        peak_score = min(100, n_significant_peaks * 10)  # More peaks = better
        band_score = (bands_with_content / len(freq_bands)) * 100
        flatness_score = (1 - spectral_flatness) * 100  # Less flat = more structured = better
        
        frequency_score = np.mean([peak_score, band_score, flatness_score])
        
        return frequency_score
    
    def calculate_data_integrity_score(self, metrics: DataQualityMetrics) -> float:
        """Calculate overall data integrity score (0-100)"""
        scores = []
        
        # Sample count score
        min_samples = self.config.get("validation", {}).get("min_samples", 1000)
        sample_score = min(100, (metrics.n_samples / min_samples) * 100) if min_samples > 0 else 100
        scores.append(sample_score)
        
        # Outlier score
        max_outlier_pct = self.config.get("validation", {}).get("max_outlier_percentage", 5.0)
        outlier_score = max(0, 100 - (metrics.outlier_percentage / max_outlier_pct) * 100)
        scores.append(outlier_score)
        
        # Sampling consistency score
        consistency_score = max(0, 100 - metrics.sampling_consistency * 1000)  # Lower CV is better
        scores.append(consistency_score)
        
        # SNR score
        min_snr = self.config.get("validation", {}).get("min_snr_db", 20)
        snr_score = min(100, (metrics.signal_snr_estimate / min_snr) * 100) if min_snr > 0 else 100
        scores.append(snr_score)
        
        # DC offset score
        max_dc_pct = self.config.get("validation", {}).get("max_dc_offset_percentage", 10)
        signal_range = metrics.signal_range[1] - metrics.signal_range[0]
        dc_offset_pct = abs(metrics.dc_offset) / signal_range * 100 if signal_range > 0 else 0
        dc_score = max(0, 100 - (dc_offset_pct / max_dc_pct) * 100)
        scores.append(dc_score)
        
        # Frequency content score
        scores.append(metrics.frequency_content_score)
        
        # Time gaps penalty
        max_gaps = 10  # Maximum acceptable number of time gaps
        gap_score = max(0, 100 - (len(metrics.time_gaps) / max_gaps) * 100)
        scores.append(gap_score)
        
        return np.mean(scores)
    
    def generate_recommendations(self, metrics: DataQualityMetrics) -> List[str]:
        """Generate recommendations based on data quality metrics"""
        recommendations = []
        
        # Sample count
        min_samples = self.config.get("validation", {}).get("min_samples", 1000)
        if metrics.n_samples < min_samples:
            recommendations.append(f"Increase sample count (current: {metrics.n_samples}, recommended: >{min_samples})")
        
        # Outliers
        max_outlier_pct = self.config.get("validation", {}).get("max_outlier_percentage", 5.0)
        if metrics.outlier_percentage > max_outlier_pct:
            recommendations.append(f"High outlier percentage ({metrics.outlier_percentage:.1f}%) - check sensor calibration and environmental conditions")
        
        # SNR
        min_snr = self.config.get("validation", {}).get("min_snr_db", 20)
        if metrics.signal_snr_estimate < min_snr:
            recommendations.append(f"Low SNR ({metrics.signal_snr_estimate:.1f} dB) - improve signal amplification or reduce noise")
        
        # DC offset
        max_dc_pct = self.config.get("validation", {}).get("max_dc_offset_percentage", 10)
        signal_range = metrics.signal_range[1] - metrics.signal_range[0]
        dc_offset_pct = abs(metrics.dc_offset) / signal_range * 100 if signal_range > 0 else 0
        if dc_offset_pct > max_dc_pct:
            recommendations.append(f"High DC offset ({dc_offset_pct:.1f}%) - check sensor calibration")
        
        # Sampling consistency
        if metrics.sampling_consistency > 0.01:  # CV > 1%
            recommendations.append("Inconsistent sampling detected - check data acquisition system timing")
        
        # Time gaps
        if len(metrics.time_gaps) > 0:
            recommendations.append(f"Found {len(metrics.time_gaps)} time gaps - check for data acquisition interruptions")
        
        # Frequency content
        if metrics.frequency_content_score < 50:
            recommendations.append("Poor frequency content - signal may be too narrow-band or noisy")
        
        # Sampling rate
        min_sr = self.config.get("validation", {}).get("min_sampling_rate", 1000)
        max_sr = self.config.get("validation", {}).get("max_sampling_rate", 100000)
        if metrics.sampling_rate < min_sr:
            recommendations.append(f"Low sampling rate ({metrics.sampling_rate:.1f} Hz) - may miss high-frequency content")
        elif metrics.sampling_rate > max_sr:
            recommendations.append(f"Very high sampling rate ({metrics.sampling_rate:.1f} Hz) - consider if necessary")
        
        if not recommendations:
            recommendations.append("Data quality is good - no major issues detected")
        
        return recommendations
    
    def validate_file(self, file_path: str) -> DataQualityMetrics:
        """Validate a single acoustic data file"""
        file_path = Path(file_path)
        
        # Basic file info
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Try to load the file
        try:
            # Try different delimiters
            data = None
            for delimiter in self.config["file_processing"]["delimiter_options"]:
                try:
                    data = pd.read_csv(file_path, header=None, 
                                     skiprows=self.config["analysis_parameters"]["skip_rows"], 
                                     delimiter=delimiter, engine='python')
                    if data.shape[1] >= 2:  # Need at least time and data columns
                        break
                except Exception:
                    continue
            
            if data is None or data.shape[1] < 2:
                raise ValueError("Could not load file or insufficient columns")
            
            # Extract time and signal data
            time_array = data.iloc[:, 0].to_numpy()
            signal_array = data.iloc[:, 1].to_numpy()
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(time_array) | np.isnan(signal_array))
            time_array = time_array[valid_mask]
            signal_array = signal_array[valid_mask]
            
            n_samples = len(signal_array)
            missing_values = len(data) - n_samples
            
            # Calculate sampling rate
            if len(time_array) > 1:
                time_diffs = np.diff(time_array)
                sampling_rate = 1.0 / np.mean(time_diffs)
                duration = time_array[-1] - time_array[0]
            else:
                sampling_rate = 0
                duration = 0
            
            # Signal statistics
            signal_mean = np.mean(signal_array)
            signal_std = np.std(signal_array)
            signal_range = (np.min(signal_array), np.max(signal_array))
            dc_offset = signal_mean
            
            # Outlier detection
            outliers, outlier_count = self.detect_outliers(signal_array)
            outlier_percentage = (outlier_count / n_samples) * 100 if n_samples > 0 else 0
            
            # Sampling consistency
            sampling_consistency, time_gaps = self.assess_sampling_consistency(time_array)
            
            # SNR estimation
            signal_snr_estimate = self.estimate_snr(signal_array)
            
            # Frequency content analysis
            frequency_content_score = self.analyze_frequency_content(signal_array, sampling_rate)
            
            # Create metrics object
            metrics = DataQualityMetrics(
                filename=file_path.name,
                file_size_mb=file_size_mb,
                n_samples=n_samples,
                sampling_rate=sampling_rate,
                duration_seconds=duration,
                signal_range=signal_range,
                signal_mean=signal_mean,
                signal_std=signal_std,
                signal_snr_estimate=signal_snr_estimate,
                missing_values=missing_values,
                outlier_count=outlier_count,
                outlier_percentage=outlier_percentage,
                time_gaps=time_gaps,
                sampling_consistency=sampling_consistency,
                dc_offset=dc_offset,
                frequency_content_score=frequency_content_score,
                data_integrity_score=0,  # Will be calculated
                recommendations=[]  # Will be generated
            )
            
            # Calculate integrity score and recommendations
            metrics.data_integrity_score = self.calculate_data_integrity_score(metrics)
            metrics.recommendations = self.generate_recommendations(metrics)
            
            logger.info(f"Validated {file_path.name}: Score {metrics.data_integrity_score:.1f}/100")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error validating {file_path.name}: {e}")
            # Return metrics with error information
            return DataQualityMetrics(
                filename=file_path.name,
                file_size_mb=file_size_mb,
                n_samples=0,
                sampling_rate=0,
                duration_seconds=0,
                signal_range=(0, 0),
                signal_mean=0,
                signal_std=0,
                signal_snr_estimate=0,
                missing_values=0,
                outlier_count=0,
                outlier_percentage=0,
                time_gaps=[],
                sampling_consistency=0,
                dc_offset=0,
                frequency_content_score=0,
                data_integrity_score=0,
                recommendations=[f"ERROR: {str(e)}"]
            )
    
    def validate_directory(self, directory_path: str) -> List[DataQualityMetrics]:
        """Validate all acoustic files in a directory"""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"Directory {directory_path} does not exist")
            return []
        
        # Find all supported files
        files = []
        for ext in self.config["file_processing"]["supported_extensions"]:
            files.extend(directory_path.glob(f'*{ext}'))
        
        if not files:
            logger.warning(f"No supported files found in {directory_path}")
            return []
        
        logger.info(f"Validating {len(files)} files in {directory_path}")
        
        results = []
        for file_path in files:
            metrics = self.validate_file(file_path)
            results.append(metrics)
        
        self.validation_results = results
        return results
    
    def create_validation_report(self, output_path: str = "validation_report.html"):
        """Create a comprehensive validation report"""
        if not self.validation_results:
            logger.warning("No validation results to report")
            return
        
        # Generate HTML report
        html_content = self._generate_html_report()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Validation report saved to {output_path}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML content for validation report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Acoustics Data Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .summary { margin: 20px 0; }
                .file-details { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                .score-high { color: green; font-weight: bold; }
                .score-medium { color: orange; font-weight: bold; }
                .score-low { color: red; font-weight: bold; }
                .recommendations { background-color: #fffacd; padding: 10px; margin: 5px 0; border-radius: 3px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Acoustics Data Validation Report</h1>
                <p>Generated on: """ + str(pd.Timestamp.now()) + """</p>
            </div>
        """
        
        # Summary statistics
        scores = [m.data_integrity_score for m in self.validation_results]
        avg_score = np.mean(scores)
        
        high_quality = sum(1 for s in scores if s >= 80)
        medium_quality = sum(1 for s in scores if 60 <= s < 80)
        low_quality = sum(1 for s in scores if s < 60)
        
        html += f"""
            <div class="summary">
                <h2>Summary</h2>
                <p>Total files analyzed: {len(self.validation_results)}</p>
                <p>Average quality score: {avg_score:.1f}/100</p>
                <p>Quality distribution:</p>
                <ul>
                    <li>High quality (≥80): {high_quality} files</li>
                    <li>Medium quality (60-79): {medium_quality} files</li>
                    <li>Low quality (<60): {low_quality} files</li>
                </ul>
            </div>
            
            <h2>File Details</h2>
        """
        
        # Individual file details
        for metrics in self.validation_results:
            score_class = "score-high" if metrics.data_integrity_score >= 80 else \
                         "score-medium" if metrics.data_integrity_score >= 60 else "score-low"
            
            html += f"""
                <div class="file-details">
                    <h3>{metrics.filename}</h3>
                    <p>Quality Score: <span class="{score_class}">{metrics.data_integrity_score:.1f}/100</span></p>
                    
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>File Size</td><td>{metrics.file_size_mb:.2f} MB</td></tr>
                        <tr><td>Samples</td><td>{metrics.n_samples:,}</td></tr>
                        <tr><td>Sampling Rate</td><td>{metrics.sampling_rate:.1f} Hz</td></tr>
                        <tr><td>Duration</td><td>{metrics.duration_seconds:.2f} seconds</td></tr>
                        <tr><td>Signal Range</td><td>{metrics.signal_range[0]:.3f} to {metrics.signal_range[1]:.3f}</td></tr>
                        <tr><td>SNR Estimate</td><td>{metrics.signal_snr_estimate:.1f} dB</td></tr>
                        <tr><td>Outliers</td><td>{metrics.outlier_percentage:.1f}% ({metrics.outlier_count:,} samples)</td></tr>
                        <tr><td>Time Gaps</td><td>{len(metrics.time_gaps)}</td></tr>
                        <tr><td>Frequency Content Score</td><td>{metrics.frequency_content_score:.1f}/100</td></tr>
                    </table>
                    
                    <div class="recommendations">
                        <strong>Recommendations:</strong>
                        <ul>
            """
            
            for rec in metrics.recommendations:
                html += f"<li>{rec}</li>"
            
            html += """
                        </ul>
                    </div>
                </div>
            """
        
        html += """
            </body>
            </html>
        """
        
        return html
    
    def save_validation_results(self, output_path: str = "validation_results.csv"):
        """Save validation results to CSV"""
        if not self.validation_results:
            logger.warning("No validation results to save")
            return
        
        # Convert to DataFrame
        data = []
        for metrics in self.validation_results:
            row = {
                'filename': metrics.filename,
                'file_size_mb': metrics.file_size_mb,
                'n_samples': metrics.n_samples,
                'sampling_rate': metrics.sampling_rate,
                'duration_seconds': metrics.duration_seconds,
                'signal_min': metrics.signal_range[0],
                'signal_max': metrics.signal_range[1],
                'signal_mean': metrics.signal_mean,
                'signal_std': metrics.signal_std,
                'snr_estimate_db': metrics.signal_snr_estimate,
                'missing_values': metrics.missing_values,
                'outlier_count': metrics.outlier_count,
                'outlier_percentage': metrics.outlier_percentage,
                'time_gaps': len(metrics.time_gaps),
                'sampling_consistency': metrics.sampling_consistency,
                'dc_offset': metrics.dc_offset,
                'frequency_content_score': metrics.frequency_content_score,
                'data_integrity_score': metrics.data_integrity_score,
                'recommendations': '; '.join(metrics.recommendations)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Validation results saved to {output_path}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Acoustics Data Validator')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to directory containing acoustic data files')
    parser.add_argument('--config', type=str, default='acoustics_config.json',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='validation_output',
                       help='Output directory for validation results')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Create validator
    validator = AcousticsDataValidator(args.config)
    
    # Validate data
    results = validator.validate_directory(args.data_path)
    
    if results:
        # Save results
        csv_path = Path(args.output_dir) / "validation_results.csv"
        html_path = Path(args.output_dir) / "validation_report.html"
        
        validator.save_validation_results(str(csv_path))
        validator.create_validation_report(str(html_path))
        
        # Print summary
        scores = [r.data_integrity_score for r in results]
        avg_score = np.mean(scores)
        
        print(f"\nValidation Complete!")
        print(f"Files analyzed: {len(results)}")
        print(f"Average quality score: {avg_score:.1f}/100")
        print(f"Results saved to: {args.output_dir}")
    else:
        print("No files were successfully validated")

if __name__ == "__main__":
    main()