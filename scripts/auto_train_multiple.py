#!/usr/bin/env python3
"""
Sistema de Entrenamiento Automático Multi-Modelo
Taï National Park Species Classification

Este script ejecuta múltiples experimentos de entrenamiento de forma secuencial
con diferentes modelos y configuraciones para encontrar la mejor combinación.

Uso:
    python scripts/auto_train_multiple.py --time-budget 6 --priority best
    python scripts/auto_train_multiple.py --quick-test  # Para probar las configuraciones
"""

import os
import sys
import argparse
import subprocess
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_utils import setup_logging


class AutoTrainer:
    """Sistema de entrenamiento automático con múltiples configuraciones."""
    
    def __init__(self, time_budget_hours: float = 8, output_dir: str = "results/auto_experiments"):
        self.time_budget = timedelta(hours=time_budget_hours)
        self.start_time = datetime.now()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(
            log_file=self.output_dir / f"auto_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            level='INFO'
        )
        
        # Fix encoding for Windows
        for handler in self.logger.handlers:
            if hasattr(handler, 'stream') and hasattr(handler.stream, 'reconfigure'):
                try:
                    handler.stream.reconfigure(encoding='utf-8')
                except:
                    pass
        
        self.completed_experiments = []
        self.failed_experiments = []
        self.results_summary = {}
        
        self.logger.info(f"AutoTrainer iniciado - Presupuesto: {time_budget_hours}h")
        self.logger.info(f"Directorio de salida: {self.output_dir}")
    
    def get_experiment_configurations(self, priority: str = "balanced") -> List[Dict[str, Any]]:
        """
        Define configuraciones de experimentos basadas en prioridad.
        
        Args:
            priority: Tipo de prioridad - 'quick', 'balanced', 'best', 'exhaustive'
        """
        
        base_config = {
            'data_dir': 'data/raw',
            'mixed_precision': True,
            'class_weights': True,
            'validation_sites': 'data/processed/validation_sites.csv'
        }
        
        if priority == "quick":
            # Configuraciones rápidas para probar
            configs = [
                {**base_config, 'model': 'resnet50', 'epochs': 2, 'batch_size': 32, 'lr': 0.001},
                {**base_config, 'model': 'efficientnet_b0', 'epochs': 2, 'batch_size': 32, 'lr': 0.001},
                {**base_config, 'model': 'efficientnet_b2', 'epochs': 2, 'batch_size': 16, 'lr': 0.0005}
            ]
        
        elif priority == "balanced":
            # Balance entre tiempo y rendimiento
            configs = [
                # ResNet variants
                {**base_config, 'model': 'resnet50', 'epochs': 30, 'batch_size': 32, 'lr': 0.001, 'focal_loss': False},
                {**base_config, 'model': 'resnet50', 'epochs': 30, 'batch_size': 32, 'lr': 0.001, 'focal_loss': True},
                
                # EfficientNet variants
                {**base_config, 'model': 'efficientnet_b2', 'epochs': 35, 'batch_size': 24, 'lr': 0.0008, 'aggressive_aug': False},
                {**base_config, 'model': 'efficientnet_b2', 'epochs': 35, 'batch_size': 24, 'lr': 0.0008, 'aggressive_aug': True},
                {**base_config, 'model': 'efficientnet_b3', 'epochs': 40, 'batch_size': 16, 'lr': 0.0005, 'focal_loss': True},
                
                # Different samplers
                {**base_config, 'model': 'efficientnet_b2', 'epochs': 30, 'batch_size': 24, 'lr': 0.0008, 'sampler': 'site_aware'},
                {**base_config, 'model': 'efficientnet_b2', 'epochs': 30, 'batch_size': 24, 'lr': 0.0008, 'sampler': 'balanced_batch'},
            ]
        
        elif priority == "best":
            # Mejores configuraciones para máximo rendimiento
            configs = [
                # EfficientNet B3 con diferentes configuraciones
                {**base_config, 'model': 'efficientnet_b3', 'epochs': 50, 'batch_size': 16, 'lr': 0.0005, 
                 'focal_loss': True, 'aggressive_aug': True, 'sampler': 'site_aware'},
                {**base_config, 'model': 'efficientnet_b3', 'epochs': 45, 'batch_size': 20, 'lr': 0.0008, 
                 'focal_loss': True, 'aggressive_aug': False, 'sampler': 'weighted'},
                
                # EfficientNet B4 (más grande)
                {**base_config, 'model': 'efficientnet_b4', 'epochs': 40, 'batch_size': 12, 'lr': 0.0003, 
                 'focal_loss': True, 'aggressive_aug': True, 'sampler': 'site_aware'},
                {**base_config, 'model': 'efficientnet_b4', 'epochs': 35, 'batch_size': 16, 'lr': 0.0005, 
                 'focal_loss': False, 'aggressive_aug': True, 'sampler': 'balanced_batch'},
                
                # ConvNeXt
                {**base_config, 'model': 'convnext_tiny', 'epochs': 35, 'batch_size': 20, 'lr': 0.0005, 
                 'focal_loss': True, 'aggressive_aug': False},
                
                # ResNet con configuraciones optimizadas
                {**base_config, 'model': 'resnet101', 'epochs': 30, 'batch_size': 24, 'lr': 0.0008, 
                 'focal_loss': True, 'sampler': 'site_aware'},
            ]
        
        elif priority == "exhaustive":
            # Búsqueda exhaustiva
            models = ['resnet50', 'resnet101', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4']
            learning_rates = [0.0003, 0.0005, 0.0008, 0.001]
            focal_loss_options = [True, False]
            aggressive_aug_options = [True, False]
            samplers = ['weighted', 'site_aware', 'balanced_batch']
            
            configs = []
            for model in models:
                for lr in learning_rates:
                    for focal in focal_loss_options:
                        for aug in aggressive_aug_options:
                            for sampler in samplers[:2]:  # Limitar combinaciones
                                batch_size = 16 if 'b4' in model else (20 if 'b3' in model else 24)
                                epochs = 35 if 'efficient' in model else 30
                                
                                configs.append({
                                    **base_config, 
                                    'model': model, 
                                    'epochs': epochs,
                                    'batch_size': batch_size, 
                                    'lr': lr, 
                                    'focal_loss': focal,
                                    'aggressive_aug': aug,
                                    'sampler': sampler
                                })
        
        # Agregar nombres de experimentos
        for i, config in enumerate(configs):
            config['experiment_name'] = self._generate_experiment_name(config, i)
        
        return configs
    
    def _generate_experiment_name(self, config: Dict[str, Any], index: int) -> str:
        """Genera un nombre descriptivo para el experimento."""
        model = config['model']
        lr = config['lr']
        focal = "focal" if config.get('focal_loss', False) else "ce"
        aug = "aggaug" if config.get('aggressive_aug', False) else "stdaug"
        sampler = config.get('sampler', 'weighted')[:3]
        
        return f"{model}_{lr}_{focal}_{aug}_{sampler}_{index:02d}"
    
    def run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta un solo experimento de entrenamiento."""
        
        experiment_name = config['experiment_name']
        self.logger.info(f"Iniciando experimento: {experiment_name}")
        
        # Construir comando
        cmd = [
            sys.executable, 'scripts/train_model.py',
            '--model', config['model'],
            '--epochs', str(config['epochs']),
            '--batch-size', str(config['batch_size']),
            '--learning-rate', str(config['lr']),
            '--experiment-name', experiment_name,
            '--mixed-precision',
            '--class-weights',
        ]
        
        # Agregar flags opcionales
        if config.get('focal_loss', False):
            cmd.append('--focal-loss')
        if config.get('aggressive_aug', False):
            cmd.append('--aggressive-aug')
        if config.get('sampler'):
            cmd.extend(['--sampler', config['sampler']])
        if config.get('validation_sites'):
            cmd.extend(['--validation-sites', config['validation_sites']])
        
        try:
            start_time = time.time()
            self.logger.info(f"Comando: {' '.join(cmd)}")
            
            # Ejecutar entrenamiento (con logs visibles)
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=False,  # Mostrar logs en tiempo real
                text=True,
                timeout=3600 * 3  # 3 horas máximo por experimento
            )
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                # Experimento exitoso
                self.logger.info(f"[OK] Experimento {experiment_name} completado en {training_time/3600:.2f}h")
                
                # Extraer métricas del resultado
                metrics = self._extract_metrics_from_output(result.stdout, experiment_name)
                
                experiment_result = {
                    'name': experiment_name,
                    'config': config,
                    'status': 'completed',
                    'training_time_hours': training_time / 3600,
                    'metrics': metrics,
                    'output_dir': f"results/models/{experiment_name}"
                }
                
                self.completed_experiments.append(experiment_result)
                return experiment_result
                
            else:
                # Experimento falló
                self.logger.error(f"[FAIL] Experimento {experiment_name} falló")
                self.logger.error(f"stdout: {result.stdout}")
                self.logger.error(f"stderr: {result.stderr}")
                
                experiment_result = {
                    'name': experiment_name,
                    'config': config,
                    'status': 'failed',
                    'error': result.stderr,
                    'training_time_hours': training_time / 3600
                }
                
                self.failed_experiments.append(experiment_result)
                return experiment_result
        
        except subprocess.TimeoutExpired:
            self.logger.error(f"[TIMEOUT] Experimento {experiment_name} excedió tiempo límite")
            return {
                'name': experiment_name,
                'config': config,
                'status': 'timeout',
                'training_time_hours': 3.0
            }
        
        except Exception as e:
            self.logger.error(f"[ERROR] Error ejecutando experimento {experiment_name}: {e}")
            self.logger.error(traceback.format_exc())
            return {
                'name': experiment_name,
                'config': config,
                'status': 'error',
                'error': str(e)
            }
    
    def _extract_metrics_from_output(self, stdout: str, experiment_name: str) -> Dict[str, float]:
        """Extrae métricas del output de entrenamiento."""
        metrics = {}
        
        try:
            # Buscar archivo de historial de entrenamiento
            history_file = project_root / f"results/models/{experiment_name}/models/training_history.json"
            
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                # Extraer mejores métricas
                if 'val_accuracy' in history and history['val_accuracy']:
                    metrics['best_val_accuracy'] = max(history['val_accuracy'])
                    metrics['final_val_accuracy'] = history['val_accuracy'][-1]
                
                if 'val_log_loss' in history and history['val_log_loss']:
                    metrics['best_val_log_loss'] = min(history['val_log_loss'])
                    metrics['final_val_log_loss'] = history['val_log_loss'][-1]
                
                if 'train_accuracy' in history and history['train_accuracy']:
                    metrics['final_train_accuracy'] = history['train_accuracy'][-1]
                
                metrics['total_epochs'] = len(history.get('val_accuracy', []))
        
        except Exception as e:
            self.logger.warning(f"No se pudieron extraer métricas para {experiment_name}: {e}")
            
            # Fallback: buscar en stdout
            lines = stdout.split('\n')
            for line in lines:
                if 'Best validation accuracy:' in line:
                    try:
                        metrics['best_val_accuracy'] = float(line.split(':')[-1].strip())
                    except:
                        pass
                elif 'Best validation loss:' in line:
                    try:
                        metrics['best_val_log_loss'] = float(line.split(':')[-1].strip())
                    except:
                        pass
        
        return metrics
    
    def has_time_remaining(self) -> bool:
        """Verifica si queda tiempo en el presupuesto."""
        elapsed = datetime.now() - self.start_time
        return elapsed < self.time_budget
    
    def get_remaining_time_hours(self) -> float:
        """Obtiene tiempo restante en horas."""
        elapsed = datetime.now() - self.start_time
        remaining = self.time_budget - elapsed
        return max(0, remaining.total_seconds() / 3600)
    
    def run_auto_training(self, priority: str = "balanced", max_experiments: Optional[int] = None) -> Dict[str, Any]:
        """
        Ejecuta entrenamiento automático con múltiples configuraciones.
        
        Args:
            priority: Prioridad de configuraciones
            max_experiments: Máximo número de experimentos (None para sin límite)
        """
        
        self.logger.info(f"Iniciando entrenamiento automático - Prioridad: {priority}")
        
        # Obtener configuraciones
        configs = self.get_experiment_configurations(priority)
        
        if max_experiments:
            configs = configs[:max_experiments]
        
        self.logger.info(f"Total de experimentos planificados: {len(configs)}")
        
        # Ejecutar experimentos
        for i, config in enumerate(configs):
            if not self.has_time_remaining():
                self.logger.info(f"[TIME] Tiempo agotado. Deteniendo después de {i} experimentos.")
                break
            
            remaining_hours = self.get_remaining_time_hours()
            remaining_experiments = len(configs) - i
            
            self.logger.info(f"Tiempo restante: {remaining_hours:.1f}h | Experimentos restantes: {remaining_experiments}")
            
            # Ejecutar experimento
            result = self.run_single_experiment(config)
            
            # Guardar progreso
            self._save_progress()
        
        # Generar resumen final
        return self._generate_final_summary()
    
    def _save_progress(self):
        """Guarda el progreso actual."""
        progress = {
            'start_time': self.start_time.isoformat(),
            'current_time': datetime.now().isoformat(),
            'time_budget_hours': self.time_budget.total_seconds() / 3600,
            'completed_experiments': self.completed_experiments,
            'failed_experiments': self.failed_experiments,
            'total_completed': len(self.completed_experiments),
            'total_failed': len(self.failed_experiments)
        }
        
        progress_file = self.output_dir / 'training_progress.json'
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2, default=str)
    
    def _generate_final_summary(self) -> Dict[str, Any]:
        """Genera resumen final de todos los experimentos."""
        
        total_time = datetime.now() - self.start_time
        
        summary = {
            'execution_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_time_hours': total_time.total_seconds() / 3600,
                'time_budget_hours': self.time_budget.total_seconds() / 3600,
                'total_experiments': len(self.completed_experiments) + len(self.failed_experiments),
                'completed_experiments': len(self.completed_experiments),
                'failed_experiments': len(self.failed_experiments),
                'success_rate': len(self.completed_experiments) / max(1, len(self.completed_experiments) + len(self.failed_experiments))
            },
            'completed_experiments': self.completed_experiments,
            'failed_experiments': self.failed_experiments
        }
        
        # Encontrar mejores experimentos
        if self.completed_experiments:
            # Mejor por accuracy
            best_acc_exp = max(
                [exp for exp in self.completed_experiments if exp.get('metrics', {}).get('best_val_accuracy')],
                key=lambda x: x['metrics']['best_val_accuracy'],
                default=None
            )
            
            # Mejor por log loss
            best_loss_exp = min(
                [exp for exp in self.completed_experiments if exp.get('metrics', {}).get('best_val_log_loss')],
                key=lambda x: x['metrics']['best_val_log_loss'],
                default=None
            )
            
            summary['best_experiments'] = {
                'best_accuracy': best_acc_exp,
                'best_log_loss': best_loss_exp
            }
            
            # Rankings
            summary['rankings'] = self._create_rankings()
        
        # Guardar resumen
        summary_file = self.output_dir / 'final_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Resumen final guardado en: {summary_file}")
        self._log_final_summary(summary)
        
        return summary
    
    def _create_rankings(self) -> Dict[str, List[Dict]]:
        """Crea rankings de experimentos por diferentes métricas."""
        
        valid_experiments = [exp for exp in self.completed_experiments 
                           if exp.get('metrics', {}).get('best_val_accuracy')]
        
        rankings = {}
        
        if valid_experiments:
            # Ranking por accuracy
            rankings['by_accuracy'] = sorted(
                valid_experiments,
                key=lambda x: x['metrics']['best_val_accuracy'],
                reverse=True
            )[:5]
            
            # Ranking por log loss
            rankings['by_log_loss'] = sorted(
                [exp for exp in valid_experiments if exp.get('metrics', {}).get('best_val_log_loss')],
                key=lambda x: x['metrics']['best_val_log_loss']
            )[:5]
            
            # Ranking por eficiencia (accuracy / tiempo)
            rankings['by_efficiency'] = sorted(
                valid_experiments,
                key=lambda x: x['metrics']['best_val_accuracy'] / max(0.1, x.get('training_time_hours', 1)),
                reverse=True
            )[:5]
        
        return rankings
    
    def _log_final_summary(self, summary: Dict[str, Any]):
        """Registra resumen final en el log."""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("RESUMEN FINAL DE ENTRENAMIENTO AUTOMÁTICO")
        self.logger.info("="*80)
        
        exec_summary = summary['execution_summary']
        self.logger.info(f"Tiempo total: {exec_summary['total_time_hours']:.2f}h / {exec_summary['time_budget_hours']:.2f}h")
        self.logger.info(f"Experimentos: {exec_summary['completed_experiments']} exitosos, {exec_summary['failed_experiments']} fallidos")
        self.logger.info(f"Tasa de éxito: {exec_summary['success_rate']:.1%}")
        
        if 'best_experiments' in summary:
            best = summary['best_experiments']
            
            if best.get('best_accuracy'):
                exp = best['best_accuracy']
                acc = exp['metrics']['best_val_accuracy']
                self.logger.info(f"\nMEJOR ACCURACY: {exp['name']}")
                self.logger.info(f"   Accuracy: {acc:.4f}")
                self.logger.info(f"   Modelo: {exp['config']['model']}")
                self.logger.info(f"   Config: lr={exp['config']['lr']}, focal={exp['config'].get('focal_loss', False)}")
            
            if best.get('best_log_loss'):
                exp = best['best_log_loss']
                loss = exp['metrics']['best_val_log_loss']
                self.logger.info(f"\nMEJOR LOG LOSS: {exp['name']}")
                self.logger.info(f"   Log Loss: {loss:.4f}")
                self.logger.info(f"   Modelo: {exp['config']['model']}")
                self.logger.info(f"   Config: lr={exp['config']['lr']}, focal={exp['config'].get('focal_loss', False)}")
        
        # Top 3 por accuracy
        if 'rankings' in summary and summary['rankings'].get('by_accuracy'):
            self.logger.info(f"\nTOP 3 POR ACCURACY:")
            for i, exp in enumerate(summary['rankings']['by_accuracy'][:3], 1):
                acc = exp['metrics']['best_val_accuracy']
                self.logger.info(f"   {i}. {exp['name']}: {acc:.4f}")
        
        self.logger.info("\n" + "="*80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Entrenamiento Automático Multi-Modelo')
    
    parser.add_argument('--time-budget', type=float, default=8.0,
                       help='Presupuesto de tiempo en horas (default: 8)')
    parser.add_argument('--priority', type=str, default='balanced',
                       choices=['quick', 'balanced', 'best', 'exhaustive'],
                       help='Prioridad de configuraciones')
    parser.add_argument('--max-experiments', type=int, default=None,
                       help='Máximo número de experimentos')
    parser.add_argument('--output-dir', type=str, default='results/auto_experiments',
                       help='Directorio de salida')
    parser.add_argument('--quick-test', action='store_true',
                       help='Prueba rápida con configuraciones mínimas')
    
    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_args()
    
    if args.quick_test:
        args.priority = 'quick'
        args.time_budget = 2.0
        args.max_experiments = 3
    
    print(f"Iniciando entrenamiento automático")
    print(f"Presupuesto de tiempo: {args.time_budget}h")
    print(f"Prioridad: {args.priority}")
    print(f"Directorio de salida: {args.output_dir}")
    
    try:
        # Crear entrenador automático
        trainer = AutoTrainer(
            time_budget_hours=args.time_budget,
            output_dir=args.output_dir
        )
        
        # Ejecutar entrenamiento
        summary = trainer.run_auto_training(
            priority=args.priority,
            max_experiments=args.max_experiments
        )
        
        print(f"\n[OK] Entrenamiento automático completado!")
        print(f"[STATS] {summary['execution_summary']['completed_experiments']} experimentos exitosos")
        print(f"[OUTPUT] Resultados en: {args.output_dir}")
        
        if summary.get('best_experiments', {}).get('best_accuracy'):
            best = summary['best_experiments']['best_accuracy']
            print(f"[BEST] Mejor modelo: {best['name']} (accuracy: {best['metrics']['best_val_accuracy']:.4f})")
    
    except KeyboardInterrupt:
        print(f"\n[STOP] Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n[ERROR] Error durante el entrenamiento: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()