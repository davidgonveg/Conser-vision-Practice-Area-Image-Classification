"""
Taï National Park - Data Preprocessing Utilities

This module provides comprehensive preprocessing utilities for camera trap images:
- Data quality checks and validation
- Image preprocessing and enhancement
- Dataset statistics and analysis
- Cache management utilities
- Data integrity verification

Key Features:
- Robust image validation
- Automatic data quality assessment
- Smart preprocessing pipelines
- Performance optimization utilities
- Comprehensive dataset analysis
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageStat, ImageEnhance
import cv2
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import json
import pickle
import hashlib
from collections import defaultdict, Counter
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
from tqdm import tqdm
import time
import shutil

logger = logging.getLogger(__name__)


class ImageValidator:
    """
    Validates camera trap images for quality and integrity.
    """
    
    def __init__(
        self,
        min_size: Tuple[int, int] = (100, 100),
        max_size: Tuple[int, int] = (10000, 10000),
        max_file_size_mb: float = 50.0,
        min_brightness: float = 10.0,
        max_brightness: float = 245.0,
        supported_formats: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
        check_corruption: bool = True
    ):
        """
        Initialize image validator.
        
        Args:
            min_size: Minimum allowed (width, height)
            max_size: Maximum allowed (width, height)
            max_file_size_mb: Maximum file size in MB
            min_brightness: Minimum average brightness
            max_brightness: Maximum average brightness
            supported_formats: Supported image formats
            check_corruption: Whether to check for corrupted images
        """
        self.min_size = min_size
        self.max_size = max_size
        self.max_file_size_mb = max_file_size_mb
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.supported_formats = [fmt.lower() for fmt in supported_formats]
        self.check_corruption = check_corruption
        
        self.validation_stats = {
            'total_checked': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'corrupted_images': 0,
            'size_issues': 0,
            'format_issues': 0,
            'brightness_issues': 0,
            'file_size_issues': 0
        }
    
    def validate_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Validate a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Validation result dictionary
        """
        result = {
            'path': str(image_path),
            'is_valid': True,
            'issues': [],
            'properties': {}
        }
        
        self.validation_stats['total_checked'] += 1
        
        try:
            # Check if file exists
            if not image_path.exists():
                result['is_valid'] = False
                result['issues'].append('file_not_found')
                return result
            
            # Check file size
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            result['properties']['file_size_mb'] = file_size_mb
            
            if file_size_mb > self.max_file_size_mb:
                result['is_valid'] = False
                result['issues'].append('file_too_large')
                self.validation_stats['file_size_issues'] += 1
            
            # Check file format
            if image_path.suffix.lower() not in self.supported_formats:
                result['is_valid'] = False
                result['issues'].append('unsupported_format')
                self.validation_stats['format_issues'] += 1
                return result
            
            # Try to open and analyze image
            if self.check_corruption:
                try:
                    with Image.open(image_path) as img:
                        # Verify image can be loaded
                        img.verify()
                    
                    # Reopen for analysis (verify() closes the image)
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')
                        
                        # Get image properties
                        width, height = img.size
                        result['properties']['width'] = width
                        result['properties']['height'] = height
                        result['properties']['channels'] = len(img.getbands())
                        
                        # Check dimensions
                        if (width < self.min_size[0] or height < self.min_size[1] or
                            width > self.max_size[0] or height > self.max_size[1]):
                            result['is_valid'] = False
                            result['issues'].append('invalid_dimensions')
                            self.validation_stats['size_issues'] += 1
                        
                        # Check brightness
                        stat = ImageStat.Stat(img)
                        avg_brightness = sum(stat.mean) / len(stat.mean)
                        result['properties']['avg_brightness'] = avg_brightness
                        
                        if (avg_brightness < self.min_brightness or 
                            avg_brightness > self.max_brightness):
                            result['is_valid'] = False
                            result['issues'].append('brightness_out_of_range')
                            self.validation_stats['brightness_issues'] += 1
                        
                        # Additional quality metrics
                        result['properties']['brightness_std'] = np.mean(stat.stddev)
                        
                except Exception as e:
                    result['is_valid'] = False
                    result['issues'].append(f'corruption_error: {str(e)}')
                    self.validation_stats['corrupted_images'] += 1
            
            # Update statistics
            if result['is_valid']:
                self.validation_stats['valid_images'] += 1
            else:
                self.validation_stats['invalid_images'] += 1
                
        except Exception as e:
            result['is_valid'] = False
            result['issues'].append(f'validation_error: {str(e)}')
            self.validation_stats['invalid_images'] += 1
        
        return result
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation statistics summary."""
        total = self.validation_stats['total_checked']
        if total == 0:
            return self.validation_stats
        
        summary = self.validation_stats.copy()
        summary['validation_rate'] = self.validation_stats['valid_images'] / total
        summary['corruption_rate'] = self.validation_stats['corrupted_images'] / total
        
        return summary


class DatasetAnalyzer:
    """
    Comprehensive analyzer for camera trap datasets.
    """
    
    def __init__(self, data_dir: Path, cache_dir: Optional[Path] = None):
        """
        Initialize dataset analyzer.
        
        Args:
            data_dir: Path to the dataset directory
            cache_dir: Directory for caching analysis results
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.image_validator = ImageValidator()
        self.analysis_cache = {}
    
    def analyze_dataset_structure(self) -> Dict[str, Any]:
        """Analyze the structure of the dataset."""
        
        structure = {
            'data_dir': str(self.data_dir),
            'files_found': {},
            'directories': [],
            'total_files': 0,
            'image_files': 0
        }
        
        # Find all files and directories
        for item in self.data_dir.rglob('*'):
            if item.is_dir():
                structure['directories'].append(str(item.relative_to(self.data_dir)))
            else:
                structure['total_files'] += 1
                suffix = item.suffix.lower()
                
                if suffix not in structure['files_found']:
                    structure['files_found'][suffix] = 0
                structure['files_found'][suffix] += 1
                
                if suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
                    structure['image_files'] += 1
        
        # Look for expected CSV files
        expected_files = ['train_features.csv', 'test_features.csv', 'train_labels.csv']
        structure['csv_files'] = {}
        
        for csv_file in expected_files:
            csv_path = self.data_dir / csv_file
            structure['csv_files'][csv_file] = {
                'exists': csv_path.exists(),
                'path': str(csv_path)
            }
            
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    structure['csv_files'][csv_file]['rows'] = len(df)
                    structure['csv_files'][csv_file]['columns'] = list(df.columns)
                except Exception as e:
                    structure['csv_files'][csv_file]['error'] = str(e)
        
        return structure
    
    def analyze_image_properties(
        self,
        sample_size: Optional[int] = None,
        use_cache: bool = True,
        n_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Analyze properties of images in the dataset.
        
        Args:
            sample_size: Number of images to sample (None for all)
            use_cache: Whether to use cached results
            n_workers: Number of worker processes
            
        Returns:
            Analysis results dictionary
        """
        
        cache_file = self.cache_dir / f"image_analysis_{sample_size or 'all'}.json"
        
        if use_cache and cache_file.exists():
            logger.info("Loading cached image analysis results")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        logger.info("Analyzing image properties...")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(self.data_dir.rglob(f'*{ext}')))
            image_paths.extend(list(self.data_dir.rglob(f'*{ext.upper()}')))
        
        if sample_size and len(image_paths) > sample_size:
            np.random.seed(42)
            image_paths = np.random.choice(image_paths, sample_size, replace=False)
        
        logger.info(f"Analyzing {len(image_paths)} images...")
        
        # Analyze images in parallel
        results = []
        
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                validation_results = list(tqdm(
                    executor.map(self.image_validator.validate_image, image_paths),
                    total=len(image_paths),
                    desc="Validating images"
                ))
        else:
            validation_results = [
                self.image_validator.validate_image(path) 
                for path in tqdm(image_paths, desc="Validating images")
            ]
        
        # Aggregate results
        valid_results = [r for r in validation_results if r['is_valid']]
        invalid_results = [r for r in validation_results if not r['is_valid']]
        
        if valid_results:
            properties = [r['properties'] for r in valid_results]
            
            analysis = {
                'total_images': len(image_paths),
                'valid_images': len(valid_results),
                'invalid_images': len(invalid_results),
                'validation_rate': len(valid_results) / len(image_paths),
                
                # Dimension statistics
                'width_stats': self._calculate_stats([p['width'] for p in properties]),
                'height_stats': self._calculate_stats([p['height'] for p in properties]),
                'aspect_ratio_stats': self._calculate_stats([
                    p['width'] / p['height'] for p in properties
                ]),
                
                # Brightness statistics
                'brightness_stats': self._calculate_stats([p['avg_brightness'] for p in properties]),
                'brightness_std_stats': self._calculate_stats([p.get('brightness_std', 0) for p in properties]),
                
                # File size statistics
                'file_size_stats': self._calculate_stats([p['file_size_mb'] for p in properties]),
                
                # Common resolutions
                'common_resolutions': self._find_common_resolutions(properties),
                
                # Issues summary
                'issues_summary': self._summarize_issues(invalid_results),
                
                # Validation summary
                'validation_summary': self.image_validator.get_validation_summary()
            }
        else:
            analysis = {
                'total_images': len(image_paths),
                'valid_images': 0,
                'invalid_images': len(invalid_results),
                'validation_rate': 0.0,
                'error': 'No valid images found',
                'issues_summary': self._summarize_issues(invalid_results)
            }
        
        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def analyze_data_consistency(self) -> Dict[str, Any]:
        """Analyze consistency between CSV files and image files."""
        
        logger.info("Analyzing data consistency...")
        
        consistency = {
            'train_consistency': {},
            'test_consistency': {},
            'overall_issues': []
        }
        
        # Check train data consistency
        train_features_path = self.data_dir / "train_features.csv"
        train_labels_path = self.data_dir / "train_labels.csv"
        
        if train_features_path.exists() and train_labels_path.exists():
            train_features = pd.read_csv(train_features_path)
            train_labels = pd.read_csv(train_labels_path)
            
            consistency['train_consistency'] = self._check_data_consistency(
                train_features, train_labels, "train"
            )
        else:
            consistency['train_consistency']['error'] = "Train CSV files not found"
        
        # Check test data consistency
        test_features_path = self.data_dir / "test_features.csv"
        
        if test_features_path.exists():
            test_features = pd.read_csv(test_features_path)
            
            consistency['test_consistency'] = self._check_data_consistency(
                test_features, None, "test"
            )
        else:
            consistency['test_consistency']['error'] = "Test CSV file not found"
        
        return consistency
    
    def _check_data_consistency(
        self, 
        features_df: pd.DataFrame, 
        labels_df: Optional[pd.DataFrame], 
        split_name: str
    ) -> Dict[str, Any]:
        """Check consistency for a specific data split."""
        
        result = {
            'csv_rows': len(features_df),
            'missing_images': [],
            'extra_images': [],
            'label_mismatches': []
        }
        
        # Check if image files exist
        for _, row in features_df.iterrows():
            image_path = self.data_dir / row['filepath']
            if not image_path.exists():
                result['missing_images'].append(row['filepath'])
        
        # Check for extra image files
        if split_name == "train":
            image_dir = self.data_dir / "train_features"
        else:
            image_dir = self.data_dir / "test_features"
        
        if image_dir.exists():
            actual_images = set(f.name for f in image_dir.glob('*.jpg'))
            expected_images = set(Path(fp).name for fp in features_df['filepath'])
            result['extra_images'] = list(actual_images - expected_images)
        
        # Check label consistency (for train data)
        if labels_df is not None:
            features_ids = set(features_df['id'])
            labels_ids = set(labels_df['id'])
            
            result['missing_labels'] = list(features_ids - labels_ids)
            result['extra_labels'] = list(labels_ids - features_ids)
            
            # Check if labels sum to 1
            class_columns = ['antelope_duiker', 'bird', 'blank', 'civet_genet', 
                           'hog', 'leopard', 'monkey_prosimian', 'rodent']
            
            if all(col in labels_df.columns for col in class_columns):
                label_sums = labels_df[class_columns].sum(axis=1)
                invalid_sums = labels_df[abs(label_sums - 1.0) > 0.01]
                
                if len(invalid_sums) > 0:
                    result['invalid_label_sums'] = invalid_sums['id'].tolist()
        
        result['images_found'] = len(features_df) - len(result['missing_images'])
        result['consistency_rate'] = result['images_found'] / len(features_df) if len(features_df) > 0 else 0
        
        return result
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures for a list of values."""
        if not values:
            return {}
        
        values = np.array(values)
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75))
        }
    
    def _find_common_resolutions(self, properties: List[Dict]) -> List[Dict]:
        """Find most common image resolutions."""
        resolutions = [(p['width'], p['height']) for p in properties]
        resolution_counts = Counter(resolutions)
        
        common_resolutions = []
        for (width, height), count in resolution_counts.most_common(10):
            common_resolutions.append({
                'width': width,
                'height': height,
                'count': count,
                'percentage': count / len(properties) * 100
            })
        
        return common_resolutions
    
    def _summarize_issues(self, invalid_results: List[Dict]) -> Dict[str, Any]:
        """Summarize validation issues."""
        issue_counts = defaultdict(int)
        
        for result in invalid_results:
            for issue in result['issues']:
                issue_counts[issue] += 1
        
        return {
            'total_invalid': len(invalid_results),
            'issue_breakdown': dict(issue_counts),
            'most_common_issue': max(issue_counts.items(), key=lambda x: x[1])[0] if issue_counts else None
        }
    
    def generate_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive dataset analysis report."""
        
        logger.info("Generating comprehensive dataset report...")
        
        report = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_structure': self.analyze_dataset_structure(),
            'image_analysis': self.analyze_image_properties(),
            'data_consistency': self.analyze_data_consistency()
        }
        
        # Add summary and recommendations
        report['summary'] = self._generate_summary(report)
        report['recommendations'] = self._generate_recommendations(report)
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Report saved to: {output_path}")
        
        return report
    
    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary from analysis report."""
        
        summary = {
            'dataset_health': 'good',  # Will be updated based on analysis
            'key_metrics': {},
            'critical_issues': []
        }
        
        # Extract key metrics
        if 'image_analysis' in report:
            img_analysis = report['image_analysis']
            summary['key_metrics'] = {
                'total_images': img_analysis.get('total_images', 0),
                'valid_images': img_analysis.get('valid_images', 0),
                'validation_rate': img_analysis.get('validation_rate', 0),
                'avg_resolution': (
                    f"{img_analysis.get('width_stats', {}).get('mean', 0):.0f}x"
                    f"{img_analysis.get('height_stats', {}).get('mean', 0):.0f}"
                ),
                'avg_file_size_mb': img_analysis.get('file_size_stats', {}).get('mean', 0)
            }
            
            # Check for critical issues
            if img_analysis.get('validation_rate', 0) < 0.95:
                summary['critical_issues'].append('Low validation rate')
                summary['dataset_health'] = 'poor'
            elif img_analysis.get('validation_rate', 0) < 0.99:
                summary['dataset_health'] = 'fair'
        
        # Check consistency issues
        if 'data_consistency' in report:
            consistency = report['data_consistency']
            
            train_consistency = consistency.get('train_consistency', {})
            if train_consistency.get('consistency_rate', 0) < 0.99:
                summary['critical_issues'].append('Train data consistency issues')
                summary['dataset_health'] = 'poor' if summary['dataset_health'] == 'good' else summary['dataset_health']
        
        return summary
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        
        recommendations = []
        
        # Image quality recommendations
        if 'image_analysis' in report:
            img_analysis = report['image_analysis']
            
            validation_rate = img_analysis.get('validation_rate', 1.0)
            if validation_rate < 0.95:
                recommendations.append(
                    f"Low validation rate ({validation_rate:.1%}). "
                    "Consider cleaning corrupted or invalid images."
                )
            
            # Resolution recommendations
            width_stats = img_analysis.get('width_stats', {})
            height_stats = img_analysis.get('height_stats', {})
            
            if width_stats.get('std', 0) > width_stats.get('mean', 1) * 0.5:
                recommendations.append(
                    "High variation in image dimensions. "
                    "Consider standardizing image sizes for training."
                )
            
            # File size recommendations
            file_size_stats = img_analysis.get('file_size_stats', {})
            if file_size_stats.get('max', 0) > 10:
                recommendations.append(
                    "Some images are very large. "
                    "Consider compressing images to reduce memory usage."
                )
        
        # Data consistency recommendations
        if 'data_consistency' in report:
            consistency = report['data_consistency']
            
            train_consistency = consistency.get('train_consistency', {})
            if train_consistency.get('missing_images'):
                recommendations.append(
                    "Some training images are missing. "
                    "Verify dataset integrity and re-download if necessary."
                )
        
        if not recommendations:
            recommendations.append("Dataset appears to be in good condition for training.")
        
        return recommendations


class CacheManager:
    """
    Manages caching for dataset operations.
    """
    
    def __init__(self, cache_dir: Path):
        """Initialize cache manager."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_cache_key(self, data: Any) -> str:
        """Generate cache key for data."""
        if isinstance(data, (str, Path)):
            # For file paths, use file modification time
            path = Path(data)
            if path.exists():
                mtime = path.stat().st_mtime
                return hashlib.md5(f"{path}_{mtime}".encode()).hexdigest()
        
        # For other data, use string representation
        return hashlib.md5(str(data).encode()).hexdigest()
    
    def cache_exists(self, cache_key: str) -> bool:
        """Check if cache exists for key."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        return cache_file.exists()
    
    def save_to_cache(self, cache_key: str, data: Any, description: str = ""):
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Update metadata
        self.metadata[cache_key] = {
            'created': time.time(),
            'description': description,
            'file_size': cache_file.stat().st_size
        }
        self._save_metadata()
    
    def load_from_cache(self, cache_key: str) -> Any:
        """Load data from cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def clear_cache(self, max_age_days: Optional[int] = None):
        """Clear old cache files."""
        current_time = time.time()
        
        for cache_key, metadata in list(self.metadata.items()):
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            should_delete = False
            
            if max_age_days:
                age_days = (current_time - metadata['created']) / (24 * 3600)
                if age_days > max_age_days:
                    should_delete = True
            
            if should_delete or not cache_file.exists():
                if cache_file.exists():
                    cache_file.unlink()
                del self.metadata[cache_key]
        
        self._save_metadata()
        logger.info(f"Cache cleared. {len(self.metadata)} entries remaining.")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(meta['file_size'] for meta in self.metadata.values())
        
        return {
            'total_entries': len(self.metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'oldest_entry': min(
                (meta['created'] for meta in self.metadata.values()), 
                default=None
            ),
            'newest_entry': max(
                (meta['created'] for meta in self.metadata.values()), 
                default=None
            )
        }


def preprocess_dataset(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    target_size: Optional[Tuple[int, int]] = None,
    quality: int = 95,
    n_workers: int = 4
) -> Dict[str, Any]:
    """
    Preprocess entire dataset for training.
    
    Args:
        data_dir: Source data directory
        output_dir: Output directory (None to overwrite)
        target_size: Target image size (width, height)
        quality: JPEG quality for compression
        n_workers: Number of worker processes
        
    Returns:
        Processing summary
    """
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir
    
    logger.info(f"Preprocessing dataset: {data_dir} -> {output_dir}")
    
    # Find all image files
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_paths.extend(list(data_dir.rglob(f'*{ext}')))
        image_paths.extend(list(data_dir.rglob(f'*{ext.upper()}')))
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Process images
    def process_image(image_path: Path) -> Dict[str, Any]:
        try:
            relative_path = image_path.relative_to(data_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                
                # Resize if needed
                if target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save with compression
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
            return {
                'success': True,
                'input_path': str(image_path),
                'output_path': str(output_path),
                'input_size': image_path.stat().st_size,
                'output_size': output_path.stat().st_size
            }
            
        except Exception as e:
            return {
                'success': False,
                'input_path': str(image_path),
                'error': str(e)
            }
    
    # Process in parallel
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(process_image, image_paths),
                total=len(image_paths),
                desc="Processing images"
            ))
    else:
        results = [process_image(path) for path in tqdm(image_paths, desc="Processing images")]
    
    # Summarize results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    total_input_size = sum(r['input_size'] for r in successful)
    total_output_size = sum(r['output_size'] for r in successful)
    
    summary = {
        'total_images': len(image_paths),
        'successful': len(successful),
        'failed': len(failed),
        'compression_ratio': total_output_size / total_input_size if total_input_size > 0 else 1.0,
        'space_saved_mb': (total_input_size - total_output_size) / (1024 * 1024),
        'failed_images': [r['input_path'] for r in failed]
    }
    
    logger.info(f"Preprocessing complete: {summary['successful']}/{summary['total_images']} images processed")
    logger.info(f"Space saved: {summary['space_saved_mb']:.1f} MB ({summary['compression_ratio']:.2f} compression ratio)")
    
    return summary


def validate_and_clean_dataset(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    remove_invalid: bool = False,
    fix_issues: bool = True
) -> Dict[str, Any]:
    """
    Validate dataset and optionally clean invalid images.
    
    Args:
        data_dir: Data directory to validate
        output_dir: Output directory for cleaned dataset
        remove_invalid: Whether to remove invalid images
        fix_issues: Whether to attempt fixing minor issues
        
    Returns:
        Validation and cleaning summary
    """
    
    data_dir = Path(data_dir)
    analyzer = DatasetAnalyzer(data_dir)
    
    logger.info("Validating and cleaning dataset...")
    
    # Run comprehensive analysis
    report = analyzer.generate_report()
    
    if not output_dir:
        output_dir = data_dir / "cleaned"
    else:
        output_dir = Path(output_dir)
    
    cleaning_summary = {
        'original_images': 0,
        'valid_images': 0,
        'fixed_images': 0,
        'removed_images': 0,
        'issues_found': []
    }
    
    # Extract validation results
    if 'image_analysis' in report:
        img_analysis = report['image_analysis']
        cleaning_summary['original_images'] = img_analysis.get('total_images', 0)
        cleaning_summary['valid_images'] = img_analysis.get('valid_images', 0)
        
        # Get detailed validation results if available
        validator = ImageValidator()
        
        # Find all images and validate them
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_paths.extend(list(data_dir.rglob(f'*{ext}')))
            image_paths.extend(list(data_dir.rglob(f'*{ext.upper()}')))
        
        valid_images = []
        invalid_images = []
        
        for image_path in tqdm(image_paths, desc="Validating images"):
            result = validator.validate_image(image_path)
            
            if result['is_valid']:
                valid_images.append(image_path)
            else:
                invalid_images.append((image_path, result['issues']))
                cleaning_summary['issues_found'].extend(result['issues'])
        
        # Process cleaning if requested
        if remove_invalid or fix_issues:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy valid images
            for img_path in tqdm(valid_images, desc="Copying valid images"):
                relative_path = img_path.relative_to(data_dir)
                output_path = output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, output_path)
            
            # Try to fix invalid images if requested
            if fix_issues:
                for img_path, issues in tqdm(invalid_images, desc="Attempting to fix images"):
                    if 'brightness_out_of_range' in issues:
                        if _fix_brightness_issue(img_path, output_dir, data_dir):
                            cleaning_summary['fixed_images'] += 1
                        else:
                            cleaning_summary['removed_images'] += 1
                    else:
                        cleaning_summary['removed_images'] += 1
            else:
                cleaning_summary['removed_images'] = len(invalid_images)
    
    # Update CSV files to reflect cleaned dataset
    if remove_invalid or fix_issues:
        _update_csv_files_for_cleaned_dataset(data_dir, output_dir, cleaning_summary)
    
    return {
        'validation_report': report,
        'cleaning_summary': cleaning_summary
    }


def _fix_brightness_issue(
    image_path: Path, 
    output_dir: Path, 
    data_dir: Path
) -> bool:
    """
    Attempt to fix brightness issues in an image.
    
    Returns:
        True if fixed successfully, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            
            # Calculate current brightness
            stat = ImageStat.Stat(img)
            current_brightness = sum(stat.mean) / len(stat.mean)
            
            # Adjust brightness to acceptable range
            if current_brightness < 50:
                # Too dark - brighten
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(1.5)
            elif current_brightness > 200:
                # Too bright - darken
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(0.7)
            
            # Save fixed image
            relative_path = image_path.relative_to(data_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, 'JPEG', quality=95)
            
            return True
            
    except Exception as e:
        logger.warning(f"Failed to fix brightness for {image_path}: {e}")
        return False


def _update_csv_files_for_cleaned_dataset(
    data_dir: Path, 
    output_dir: Path, 
    cleaning_summary: Dict[str, Any]
):
    """Update CSV files to match cleaned dataset."""
    
    # Get list of remaining images
    remaining_images = set()
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        for img_path in output_dir.rglob(f'*{ext}'):
            relative_path = img_path.relative_to(output_dir)
            remaining_images.add(str(relative_path))
        for img_path in output_dir.rglob(f'*{ext.upper()}'):
            relative_path = img_path.relative_to(output_dir)
            remaining_images.add(str(relative_path))
    
    # Update train_features.csv
    train_features_path = data_dir / "train_features.csv"
    if train_features_path.exists():
        df = pd.read_csv(train_features_path)
        original_count = len(df)
        df_cleaned = df[df['filepath'].isin(remaining_images)]
        
        output_path = output_dir / "train_features.csv"
        df_cleaned.to_csv(output_path, index=False)
        
        logger.info(f"Updated train_features.csv: {len(df_cleaned)}/{original_count} rows kept")
    
    # Update train_labels.csv
    train_labels_path = data_dir / "train_labels.csv"
    if train_labels_path.exists():
        train_features_cleaned = pd.read_csv(output_dir / "train_features.csv")
        valid_ids = set(train_features_cleaned['id'])
        
        df_labels = pd.read_csv(train_labels_path)
        df_labels_cleaned = df_labels[df_labels['id'].isin(valid_ids)]
        
        output_path = output_dir / "train_labels.csv"
        df_labels_cleaned.to_csv(output_path, index=False)
        
        logger.info(f"Updated train_labels.csv: {len(df_labels_cleaned)}/{len(df_labels)} rows kept")
    
    # Update test_features.csv
    test_features_path = data_dir / "test_features.csv"
    if test_features_path.exists():
        df = pd.read_csv(test_features_path)
        original_count = len(df)
        df_cleaned = df[df['filepath'].isin(remaining_images)]
        
        output_path = output_dir / "test_features.csv"
        df_cleaned.to_csv(output_path, index=False)
        
        logger.info(f"Updated test_features.csv: {len(df_cleaned)}/{original_count} rows kept")


# Utility functions for specific preprocessing tasks
def create_balanced_subset(
    data_dir: Path,
    output_dir: Path,
    samples_per_class: int = 1000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Create a balanced subset of the dataset for quick experimentation.
    
    Args:
        data_dir: Source data directory
        output_dir: Output directory for subset
        samples_per_class: Number of samples per class
        random_seed: Random seed for reproducibility
        
    Returns:
        Subset creation summary
    """
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_features = pd.read_csv(data_dir / "train_features.csv")
    train_labels = pd.read_csv(data_dir / "train_labels.csv")
    
    # Merge for easier processing
    data = train_features.merge(train_labels, on='id')
    
    # Sample from each class
    np.random.seed(random_seed)
    subset_data = []
    
    class_columns = ['antelope_duiker', 'bird', 'blank', 'civet_genet', 
                     'hog', 'leopard', 'monkey_prosimian', 'rodent']
    
    summary = {'samples_per_class': {}, 'total_samples': 0}
    
    for class_col in class_columns:
        class_data = data[data[class_col] == 1]
        n_available = len(class_data)
        n_sample = min(samples_per_class, n_available)
        
        if n_sample > 0:
            sampled = class_data.sample(n=n_sample, random_state=random_seed)
            subset_data.append(sampled)
            summary['samples_per_class'][class_col] = n_sample
        else:
            summary['samples_per_class'][class_col] = 0
    
    # Combine all samples
    if subset_data:
        subset_df = pd.concat(subset_data, ignore_index=True)
        summary['total_samples'] = len(subset_df)
        
        # Save subset CSV files
        subset_features = subset_df[train_features.columns]
        subset_labels = subset_df[train_labels.columns]
        
        subset_features.to_csv(output_dir / "train_features.csv", index=False)
        subset_labels.to_csv(output_dir / "train_labels.csv", index=False)
        
        # Copy image files
        train_img_dir = output_dir / "train_features"
        train_img_dir.mkdir(exist_ok=True)
        
        for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Copying images"):
            src_path = data_dir / row['filepath']
            dst_path = output_dir / row['filepath']
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
        
        logger.info(f"Created balanced subset: {summary['total_samples']} samples")
        
    return summary


# Example usage and testing
if __name__ == "__main__":
    from pathlib import Path
    
    # Test preprocessing utilities
    data_dir = Path("data/raw")
    
    if data_dir.exists():
        print("Testing preprocessing utilities...")
        
        try:
            # Test dataset analyzer
            analyzer = DatasetAnalyzer(data_dir)
            
            print("✅ DatasetAnalyzer created")
            
            # Test structure analysis
            structure = analyzer.analyze_dataset_structure()
            print(f"✅ Structure analysis: {structure['total_files']} files found")
            
            # Test image analysis (small sample)
            img_analysis = analyzer.analyze_image_properties(sample_size=10)
            print(f"✅ Image analysis: {img_analysis.get('valid_images', 0)} valid images")
            
            # Test data consistency
            consistency = analyzer.analyze_data_consistency()
            print("✅ Consistency analysis completed")
            
            # Test cache manager
            cache_manager = CacheManager(data_dir / "cache")
            cache_stats = cache_manager.get_cache_stats()
            print(f"✅ Cache manager: {cache_stats['total_entries']} entries")
            
            # Test validation
            validator = ImageValidator()
            print("✅ ImageValidator created")
            
        except Exception as e:
            print(f"❌ Preprocessing test failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"Data directory not found: {data_dir}")
        print("This is normal if running outside the project directory.")
    
    print("\n🎉 Preprocessing module tests completed!")
    print("\nKey features implemented:")
    print("  🔍 Comprehensive image validation")
    print("  📊 Dataset structure analysis")
    print("  🧹 Automatic data cleaning")
    print("  💾 Smart caching system")
    print("  📋 Detailed reporting")
    print("  ⚖️  Balanced subset creation")
    print("  🔧 Batch preprocessing utilities")