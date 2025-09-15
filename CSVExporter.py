import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class CSVExporter:
    """Exportiert Detection-Ergebnisse in CSV-Dateien"""
    
    def __init__(self, output_dir: str = "./csv_exports"):
        """
        Initialisiert den CSV-Exporter
        
        Args:
            output_dir: Verzeichnis für CSV-Exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # CSV-Feldnamen für Konsistenz
        self.run_headers = [
            'run_id', 'model_name', 'model_version', 'start_time', 'end_time',
            'total_images', 'successful_detections', 'failed_detections',
            'avg_processing_time', 'total_processing_time', 'avg_cpu_usage',
            'max_cpu_usage', 'avg_memory_usage', 'max_memory_usage',
            'avg_gpu_usage', 'max_gpu_usage', 'status', 'error_message',
            'config_json'
        ]
        
        self.result_headers = [
            'run_id', 'image_path', 'image_filename', 'classification',
            'processing_time', 'success', 'persons_detected', 'avg_confidence',
            'max_confidence', 'min_confidence', 'is_uncertain', 'error_message',
            'confidence_scores', 'model_output_json', 'timestamp'
        ]
        
    def create_run_csv(self, run_id: str, model_name: str) -> tuple[Path, Path]:
        """
        Erstellt CSV-Dateien für einen neuen Run
        
        Args:
            run_id: Eindeutige Run-ID
            model_name: Name des verwendeten Modells
            
        Returns:
            Tuple aus (run_info_csv_path, results_csv_path)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_model_name = self._sanitize_filename(model_name)
        
        # Dateinamen generieren
        run_info_filename = f"{timestamp}_{safe_model_name}_run_{run_id[:8]}.csv"
        results_filename = f"{timestamp}_{safe_model_name}_results_{run_id[:8]}.csv"
        
        run_info_path = self.output_dir / run_info_filename
        results_path = self.output_dir / results_filename
        
        # Run-Info CSV erstellen (wird später gefüllt)
        with open(run_info_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.run_headers)
            writer.writeheader()
        
        # Results CSV erstellen
        with open(results_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.result_headers)
            writer.writeheader()
        
        print(f"✓ CSV-Dateien erstellt:")
        print(f"  Run-Info: {run_info_path}")
        print(f"  Ergebnisse: {results_path}")
        
        return run_info_path, results_path
    
    def write_result(self, csv_path: Path, result_data: Dict[str, Any]):
        """
        Schreibt ein Erkennungsergebnis in die Results-CSV
        
        Args:
            csv_path: Pfad zur Results-CSV
            result_data: Dictionary mit Ergebnis-Daten
        """
        try:
            # Daten für CSV vorbereiten
            csv_row = self._prepare_result_row(result_data)
            
            # An CSV anhängen
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.result_headers)
                writer.writerow(csv_row)
                
        except Exception as e:
            print(f"⚠ Fehler beim Schreiben in CSV: {e}")
    
    def write_run_info(self, csv_path: Path, run_data: Dict[str, Any]):
        """
        Schreibt Run-Informationen in die Run-Info CSV
        
        Args:
            csv_path: Pfad zur Run-Info CSV
            run_data: Dictionary mit Run-Daten
        """
        try:
            # Daten für CSV vorbereiten
            csv_row = self._prepare_run_row(run_data)
            
            # CSV überschreiben (da nur ein Run pro Datei)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.run_headers)
                writer.writeheader()
                writer.writerow(csv_row)
                
        except Exception as e:
            print(f"⚠ Fehler beim Schreiben der Run-Info in CSV: {e}")
    
    def _prepare_result_row(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Bereitet Ergebnis-Daten für CSV-Export vor"""
        model_output = data.get('model_output')
        
        return {
            'run_id': data.get('run_id', ''),
            'image_path': data.get('image_path', ''),
            'image_filename': data.get('image_filename', ''),
            'classification': data.get('classification', ''),
            'processing_time': self._safe_float(data.get('processing_time')),
            'success': self._safe_bool_string(data.get('success', True)),
            'persons_detected': self._safe_int(model_output.get('persons_detected', 0) if model_output else 0),
            'avg_confidence': self._safe_float(model_output.get('avg_confidence') if model_output else None),
            'max_confidence': self._safe_float(model_output.get('max_confidence') if model_output else None),
            'min_confidence': self._safe_float(model_output.get('min_confidence') if model_output else None),
            'is_uncertain': self._safe_bool_string(model_output.get('uncertain', False) if model_output else False),
            'error_message': data.get('error_message', ''),
            'confidence_scores': data.get('confidence_scores', ''),
            'model_output_json': json.dumps(model_output, ensure_ascii=False) if model_output else '',
            'timestamp': datetime.now().isoformat()
        }
    
    def _prepare_run_row(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Bereitet Run-Daten für CSV-Export vor"""
        return {
            'run_id': data.get('run_id', ''),
            'model_name': data.get('model_name', ''),
            'model_version': data.get('model_version', ''),
            'start_time': self._safe_datetime_string(data.get('start_time')),
            'end_time': self._safe_datetime_string(data.get('end_time')),
            'total_images': self._safe_int(data.get('total_images')),
            'successful_detections': self._safe_int(data.get('successful_detections')),
            'failed_detections': self._safe_int(data.get('failed_detections')),
            'avg_processing_time': self._safe_float(data.get('avg_processing_time')),
            'total_processing_time': self._safe_float(data.get('total_processing_time')),
            'avg_cpu_usage': self._safe_float(data.get('system_stats', {}).get('avg_cpu')),
            'max_cpu_usage': self._safe_float(data.get('system_stats', {}).get('max_cpu')),
            'avg_memory_usage': self._safe_float(data.get('system_stats', {}).get('avg_memory')),
            'max_memory_usage': self._safe_float(data.get('system_stats', {}).get('max_memory')),
            'avg_gpu_usage': self._safe_float(data.get('system_stats', {}).get('avg_gpu')),
            'max_gpu_usage': self._safe_float(data.get('system_stats', {}).get('max_gpu')),
            'status': data.get('status', ''),
            'error_message': data.get('error_message', ''),
            'config_json': json.dumps(data.get('config', {}), ensure_ascii=False)
        }
    
    def _sanitize_filename(self, filename: str) -> str:
        """Entfernt problematische Zeichen aus Dateinamen"""
        import re
        # Ersetze problematische Zeichen
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = re.sub(r'\s+', '_', sanitized)
        return sanitized[:50]  # Länge begrenzen
    
    def _safe_float(self, value) -> str:
        """Sichere Konvertierung zu Float-String für CSV"""
        if value is None:
            return ''
        try:
            return str(float(value))
        except (ValueError, TypeError):
            return ''
    
    def _safe_int(self, value) -> str:
        """Sichere Konvertierung zu Int-String für CSV"""
        if value is None:
            return ''
        try:
            return str(int(value))
        except (ValueError, TypeError):
            return ''
    
    def _safe_bool_string(self, value) -> str:
        """Konvertiert Boolean zu String für CSV"""
        if value is None:
            return ''
        return 'true' if bool(value) else 'false'
    
    def _safe_datetime_string(self, value) -> str:
        """Konvertiert Datetime zu ISO-String für CSV"""
        if value is None:
            return ''
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)
    
    def get_export_summary(self) -> Dict[str, Any]:
        """
        Gibt eine Übersicht über alle CSV-Exports zurück
        
        Returns:
            Dictionary mit Export-Statistiken
        """
        csv_files = list(self.output_dir.glob("*.csv"))
        
        run_files = [f for f in csv_files if '_run_' in f.name]
        result_files = [f for f in csv_files if '_results_' in f.name]
        
        return {
            'export_directory': str(self.output_dir),
            'total_csv_files': len(csv_files),
            'run_info_files': len(run_files),
            'result_files': len(result_files),
            'latest_files': [f.name for f in sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]]
        }