import time
import psutil
import threading
from typing import Dict
import numpy as np



class SystemMonitor:
    """Überwacht Systemressourcen während der Verarbeitung"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self._lock = threading.Lock()
        
    def start_monitoring(self):
        """Startet das Monitoring in einem separaten Thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stoppt das Monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Monitoring Loop - läuft alle 0.5 Sekunden"""
        while self.monitoring:
            with self._lock:
                # CPU Usage
                self.cpu_usage.append(psutil.cpu_percent())
                
                # Memory Usage
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)
                
                # GPU Usage (falls verfügbar)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.gpu_usage.append(gpus[0].load * 100)
                    else:
                        self.gpu_usage.append(0)
                except ImportError:
                    self.gpu_usage.append(0)
                    
            time.sleep(0.5)
            
    def get_average_usage(self) -> Dict[str, float]:
        """Gibt durchschnittliche Ressourcennutzung zurück"""
        with self._lock:
            return {
                'avg_cpu': np.mean(self.cpu_usage) if self.cpu_usage else 0,
                'max_cpu': max(self.cpu_usage) if self.cpu_usage else 0,
                'avg_memory': np.mean(self.memory_usage) if self.memory_usage else 0,
                'max_memory': max(self.memory_usage) if self.memory_usage else 0,
                'avg_gpu': np.mean(self.gpu_usage) if self.gpu_usage else 0,
                'max_gpu': max(self.gpu_usage) if self.gpu_usage else 0
            }
    
    def reset_stats(self):
        """Setzt alle Statistiken zurück"""
        with self._lock:
            self.cpu_usage.clear()
            self.memory_usage.clear()
            self.gpu_usage.clear()
