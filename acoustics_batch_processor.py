#!/usr/bin/env python3
"""
Acoustics Batch Processor
Comprehensive batch processing system for acoustic analysis workflows.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import subprocess
import shutil
from dataclasses import dataclass, asdict
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingJob:
    """Represents a single processing job"""
    job_id: str
    data_path: str
    output_path: str
    config_path: str
    job_type: str  # 'analysis', 'validation', 'both'
    priority: int = 0
    status: str = 'pending'  # pending, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    results_summary: Optional[Dict] = None

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    max_concurrent_jobs: int = 2
    retry_failed_jobs: bool = True
    max_retries: int = 3
    cleanup_temp_files: bool = True
    send_notifications: bool = False
    notification_email: Optional[str] = None
    log_level: str = 'INFO'
    timeout_minutes: int = 60

class AcousticsBatchProcessor:
    """Batch processor for acoustic analysis workflows"""
    
    def __init__(self, config_path: str = "batch_config.json"):
        self.config = self.load_batch_config(config_path)
        self.job_queue: List[ProcessingJob] = []
        self.running_jobs: List[ProcessingJob] = []
        self.completed_jobs: List[ProcessingJob] = []
        self.failed_jobs: List[ProcessingJob] = []
        
        # Setup directories
        self.setup_directories()
        
    def load_batch_config(self, config_path: str) -> BatchConfig:
        """Load batch processing configuration"""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return BatchConfig(**config_dict)
        except FileNotFoundError:
            logger.warning(f"Batch config file {config_path} not found, using defaults")
            return BatchConfig()
        except Exception as e:
            logger.error(f"Error loading batch config: {e}")
            return BatchConfig()
    
    def setup_directories(self):
        """Setup necessary directories"""
        directories = [
            "batch_output",
            "batch_logs",
            "batch_temp",
            "batch_reports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def create_job(self, data_path: str, output_path: str, config_path: str, 
                   job_type: str = 'analysis', priority: int = 0) -> str:
        """Create a new processing job"""
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.job_queue)}"
        
        job = ProcessingJob(
            job_id=job_id,
            data_path=data_path,
            output_path=output_path,
            config_path=config_path,
            job_type=job_type,
            priority=priority
        )
        
        self.job_queue.append(job)
        logger.info(f"Created job {job_id}: {job_type} for {data_path}")
        
        return job_id
    
    def add_directory_jobs(self, base_data_path: str, job_type: str = 'both', 
                          auto_organize: bool = True) -> List[str]:
        """Add jobs for all subdirectories in a base path"""
        base_path = Path(base_data_path)
        
        if not base_path.exists():
            logger.error(f"Base data path {base_data_path} does not exist")
            return []
        
        job_ids = []
        
        # Find all subdirectories with acoustic data
        for subdir in base_path.iterdir():
            if subdir.is_dir():
                # Check if directory contains acoustic data files
                acoustic_files = []
                for ext in ['.csv', '.txt', '.dat']:
                    acoustic_files.extend(list(subdir.glob(f'*{ext}')))
                
                if acoustic_files:
                    # Create job for this directory
                    if auto_organize:
                        output_path = f"batch_output/{subdir.name}"
                    else:
                        output_path = f"batch_output/{subdir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    job_id = self.create_job(
                        data_path=str(subdir),
                        output_path=output_path,
                        config_path="acoustics_config.json",
                        job_type=job_type
                    )
                    job_ids.append(job_id)
        
        logger.info(f"Created {len(job_ids)} jobs from {base_data_path}")
        return job_ids
    
    def run_validation_job(self, job: ProcessingJob) -> bool:
        """Run data validation job"""
        try:
            # Create output directory
            Path(job.output_path).mkdir(parents=True, exist_ok=True)
            
            # Run validation script
            cmd = [
                sys.executable, "acoustics_data_validator.py",
                "--data-path", job.data_path,
                "--config", job.config_path,
                "--output-dir", f"{job.output_path}/validation"
            ]
            
            logger.info(f"Running validation for job {job.job_id}")
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=self.config.timeout_minutes * 60)
            
            if result.returncode == 0:
                logger.info(f"Validation completed for job {job.job_id}")
                return True
            else:
                logger.error(f"Validation failed for job {job.job_id}: {result.stderr}")
                job.error_message = result.stderr
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Validation job {job.job_id} timed out")
            job.error_message = "Job timed out"
            return False
        except Exception as e:
            logger.error(f"Error running validation job {job.job_id}: {e}")
            job.error_message = str(e)
            return False
    
    def run_analysis_job(self, job: ProcessingJob) -> bool:
        """Run acoustic analysis job"""
        try:
            # Create output directory
            Path(job.output_path).mkdir(parents=True, exist_ok=True)
            
            # Copy the main analysis script to output directory for reference
            analysis_script = "FFT_sound_txt"
            if Path(analysis_script).exists():
                shutil.copy2(analysis_script, f"{job.output_path}/analysis_script_used.py")
            
            # Run analysis by importing and executing the main script
            # This is a simplified approach - in practice, you'd want to refactor 
            # the main script into a callable function
            logger.info(f"Running analysis for job {job.job_id}")
            
            # Save current working directory
            original_cwd = os.getcwd()
            
            try:
                # Change to output directory to contain output files
                os.chdir(job.output_path)
                
                # Create a modified version of the analysis script for this job
                self.create_job_specific_analysis_script(job)
                
                # Run the analysis script
                result = subprocess.run([sys.executable, "job_analysis_script.py"], 
                                      capture_output=True, text=True,
                                      timeout=self.config.timeout_minutes * 60)
                
                if result.returncode == 0:
                    logger.info(f"Analysis completed for job {job.job_id}")
                    return True
                else:
                    logger.error(f"Analysis failed for job {job.job_id}: {result.stderr}")
                    job.error_message = result.stderr
                    return False
                    
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            logger.error(f"Analysis job {job.job_id} timed out")
            job.error_message = "Job timed out"
            return False
        except Exception as e:
            logger.error(f"Error running analysis job {job.job_id}: {e}")
            job.error_message = str(e)
            return False
    
    def create_job_specific_analysis_script(self, job: ProcessingJob):
        """Create a job-specific analysis script"""
        # Read the original analysis script
        with open("FFT_sound_txt", 'r') as f:
            original_script = f.read()
        
        # Modify the path to point to the job's data directory
        modified_script = original_script.replace(
            "path = r'D:\\OneDrive - Arad Technologies Ltd\\ARAD_Projects\\ALD\\tests\\test_10072025\\test_10072025\\txt_files'",
            f"path = r'{job.data_path}'"
        )
        
        # Write the modified script
        job_script_path = Path(job.output_path) / "job_analysis_script.py"
        with open(job_script_path, 'w') as f:
            f.write(modified_script)
    
    def process_job(self, job: ProcessingJob) -> bool:
        """Process a single job"""
        job.status = 'running'
        job.start_time = datetime.now().isoformat()
        
        success = True
        
        try:
            if job.job_type in ['validation', 'both']:
                if not self.run_validation_job(job):
                    success = False
            
            if job.job_type in ['analysis', 'both'] and success:
                if not self.run_analysis_job(job):
                    success = False
            
            job.end_time = datetime.now().isoformat()
            
            if success:
                job.status = 'completed'
                self.completed_jobs.append(job)
                logger.info(f"Job {job.job_id} completed successfully")
            else:
                job.status = 'failed'
                self.failed_jobs.append(job)
                logger.error(f"Job {job.job_id} failed")
            
            return success
            
        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            job.end_time = datetime.now().isoformat()
            self.failed_jobs.append(job)
            logger.error(f"Job {job.job_id} failed with exception: {e}")
            return False
    
    def run_batch(self) -> Dict:
        """Run all jobs in the batch queue"""
        logger.info(f"Starting batch processing with {len(self.job_queue)} jobs")
        start_time = datetime.now()
        
        # Sort jobs by priority (higher priority first)
        self.job_queue.sort(key=lambda x: x.priority, reverse=True)
        
        while self.job_queue or self.running_jobs:
            # Start new jobs if we have capacity
            while (len(self.running_jobs) < self.config.max_concurrent_jobs and 
                   self.job_queue):
                job = self.job_queue.pop(0)
                self.running_jobs.append(job)
                
                # In a real implementation, you'd run this in a separate thread/process
                self.process_job(job)
                self.running_jobs.remove(job)
            
            # Small delay to prevent busy waiting
            time.sleep(1)
        
        end_time = datetime.now()
        
        # Generate batch summary
        summary = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_minutes': (end_time - start_time).total_seconds() / 60,
            'total_jobs': len(self.completed_jobs) + len(self.failed_jobs),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'success_rate': len(self.completed_jobs) / (len(self.completed_jobs) + len(self.failed_jobs)) * 100 if (self.completed_jobs or self.failed_jobs) else 0
        }
        
        logger.info(f"Batch processing completed: {summary['completed_jobs']}/{summary['total_jobs']} jobs successful")
        
        # Save batch report
        self.save_batch_report(summary)
        
        return summary
    
    def save_batch_report(self, summary: Dict):
        """Save comprehensive batch processing report"""
        report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"batch_reports/batch_report_{report_time}.json"
        
        # Compile full report
        report = {
            'summary': summary,
            'completed_jobs': [asdict(job) for job in self.completed_jobs],
            'failed_jobs': [asdict(job) for job in self.failed_jobs],
            'configuration': asdict(self.config)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Batch report saved to {report_path}")
        
        # Also create a summary CSV
        self.create_summary_csv(report_time)
    
    def create_summary_csv(self, report_time: str):
        """Create a CSV summary of all jobs"""
        import pandas as pd
        
        all_jobs = self.completed_jobs + self.failed_jobs
        
        if not all_jobs:
            return
        
        # Convert to DataFrame
        data = []
        for job in all_jobs:
            row = {
                'job_id': job.job_id,
                'data_path': job.data_path,
                'output_path': job.output_path,
                'job_type': job.job_type,
                'status': job.status,
                'start_time': job.start_time,
                'end_time': job.end_time,
                'duration_minutes': self.calculate_job_duration(job),
                'error_message': job.error_message or ''
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        csv_path = f"batch_reports/batch_summary_{report_time}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Batch summary CSV saved to {csv_path}")
    
    def calculate_job_duration(self, job: ProcessingJob) -> Optional[float]:
        """Calculate job duration in minutes"""
        if job.start_time and job.end_time:
            start = datetime.fromisoformat(job.start_time)
            end = datetime.fromisoformat(job.end_time)
            return (end - start).total_seconds() / 60
        return None
    
    def retry_failed_jobs(self) -> int:
        """Retry failed jobs"""
        if not self.config.retry_failed_jobs:
            return 0
        
        retried_count = 0
        
        # Copy failed jobs to retry (to avoid modifying list while iterating)
        jobs_to_retry = self.failed_jobs.copy()
        
        for job in jobs_to_retry:
            # Check if job has exceeded max retries
            retry_count = getattr(job, 'retry_count', 0)
            if retry_count < self.config.max_retries:
                # Reset job status and add back to queue
                job.status = 'pending'
                job.start_time = None
                job.end_time = None
                job.error_message = None
                job.retry_count = retry_count + 1
                
                self.job_queue.append(job)
                self.failed_jobs.remove(job)
                retried_count += 1
                
                logger.info(f"Retrying job {job.job_id} (attempt {job.retry_count})")
        
        return retried_count
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        if self.config.cleanup_temp_files:
            temp_dir = Path("batch_temp")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                temp_dir.mkdir()
                logger.info("Cleaned up temporary files")
    
    def get_status_summary(self) -> Dict:
        """Get current status summary"""
        return {
            'queued_jobs': len(self.job_queue),
            'running_jobs': len(self.running_jobs),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'total_jobs': len(self.job_queue) + len(self.running_jobs) + len(self.completed_jobs) + len(self.failed_jobs)
        }

def create_sample_batch_config():
    """Create a sample batch configuration file"""
    config = {
        "max_concurrent_jobs": 2,
        "retry_failed_jobs": True,
        "max_retries": 3,
        "cleanup_temp_files": True,
        "send_notifications": False,
        "notification_email": None,
        "log_level": "INFO",
        "timeout_minutes": 60
    }
    
    with open("batch_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Created sample batch configuration file: batch_config.json")

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Acoustics Batch Processor')
    parser.add_argument('--config', type=str, default='batch_config.json',
                       help='Path to batch configuration file')
    parser.add_argument('--create-config', action='store_true',
                       help='Create sample batch configuration file')
    parser.add_argument('--data-path', type=str,
                       help='Path to directory containing acoustic data')
    parser.add_argument('--job-type', choices=['analysis', 'validation', 'both'], 
                       default='both', help='Type of processing job')
    parser.add_argument('--auto-organize', action='store_true',
                       help='Automatically organize subdirectories as separate jobs')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_batch_config()
        return
    
    if not args.data_path:
        logger.error("Data path is required")
        return
    
    # Create batch processor
    processor = AcousticsBatchProcessor(args.config)
    
    # Add jobs
    if args.auto_organize:
        job_ids = processor.add_directory_jobs(args.data_path, args.job_type)
        logger.info(f"Added {len(job_ids)} jobs")
    else:
        job_id = processor.create_job(
            data_path=args.data_path,
            output_path=f"batch_output/{Path(args.data_path).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config_path="acoustics_config.json",
            job_type=args.job_type
        )
        logger.info(f"Added job {job_id}")
    
    # Run batch processing
    if processor.job_queue:
        summary = processor.run_batch()
        
        # Print final summary
        print(f"\nBatch Processing Complete!")
        print(f"Jobs completed: {summary['completed_jobs']}")
        print(f"Jobs failed: {summary['failed_jobs']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Total duration: {summary['duration_minutes']:.1f} minutes")
        
        # Retry failed jobs if configured
        if processor.failed_jobs and processor.config.retry_failed_jobs:
            retried = processor.retry_failed_jobs()
            if retried > 0:
                logger.info(f"Retrying {retried} failed jobs")
                retry_summary = processor.run_batch()
                print(f"\nRetry batch completed:")
                print(f"Additional jobs completed: {retry_summary['completed_jobs']}")
        
        # Cleanup
        processor.cleanup_temp_files()
    else:
        print("No jobs were created")

if __name__ == "__main__":
    main()