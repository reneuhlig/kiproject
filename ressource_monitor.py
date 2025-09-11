#!/usr/bin/env python3
"""
System Resource Monitoring Extension
Erweitert die AI-Modell Scripts um System-Ressourcen-Tracking

Neue Features:
- CPU/Memory/GPU Usage pro Inferenz
- System Load Monitoring
- Disk I/O Tracking
- Network Usage (optional)
- Performance Metriken in separater Tabelle

Installation:
  pip install psutil GPUtil nvidia-ml-py3
  # nvidia-ml-py3 nur für NVIDIA GPU Monitoring
"""

import os
import time
import json
import datetime as dt
import threading
from typing import Dict, Optional, List
import psutil
import mysql.connector
from dateutil.tz import tzlocal

# GPU Monitoring (optional)
try:
    import GPUtil
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# =========================
# System Monitoring Konfiguration
# =========================
MONITOR_GPU = os.getenv("MONITOR_GPU", "true").lower() == "true" and GPU_AVAILABLE
MONITOR_NETWORK = os.getenv("MONITOR_NETWORK", "false").lower() == "true"
MONITOR_DISK_IO = os.getenv("MONITOR_DISK_IO", "true").lower() == "true"
SAMPLING_INTERVAL = float(os.getenv("SAMPLING_INTERVAL", "0.1"))  # Sekunden zwischen Messungen

# =========================
# Erweiterte DB-Tabellen
# =========================
SYSTEM_MONITORING_TABLES = [
    # System Performance Metrics
    """
    CREATE TABLE IF NOT EXISTS System_Performance (
        perf_id INT AUTO_INCREMENT PRIMARY KEY,
        result_id INT NULL,
        image_id INT NULL,
        model_id INT NULL,
        timestamp DATETIME NOT NULL,
        
        -- CPU Metriken
        cpu_percent FLOAT NULL,
        cpu_count_logical INT NULL,
        cpu_count_physical INT NULL,
        cpu_freq_current FLOAT NULL,
        cpu_freq_max FLOAT NULL,
        
        -- Memory Metriken
        memory_total BIGINT NULL,
        memory_available BIGINT NULL,
        memory_used BIGINT NULL,
        memory_percent FLOAT NULL,
        memory_swap_total BIGINT NULL,
        memory_swap_used BIGINT NULL,
        
        -- GPU Metriken (falls verfügbar)
        gpu_count INT NULL,
        gpu_memory_total BIGINT NULL,
        gpu_memory_used BIGINT NULL,
        gpu_utilization FLOAT NULL,
        gpu_temperature FLOAT NULL,
        gpu_power_draw FLOAT NULL,
        
        -- Disk I/O
        disk_read_bytes BIGINT NULL,
        disk_write_bytes BIGINT NULL,
        disk_read_count INT NULL,
        disk_write_count INT NULL,
        
        -- Network I/O (optional)
        network_bytes_sent BIGINT NULL,
        network_bytes_recv BIGINT NULL,
        network_packets_sent BIGINT NULL,
        network_packets_recv BIGINT NULL,
        
        -- Process-spezifische Metriken
        process_cpu_percent FLOAT NULL,
        process_memory_rss BIGINT NULL,
        process_memory_vms BIGINT NULL,
        process_num_threads INT NULL,
        process_num_fds INT NULL,
        
        CONSTRAINT fk_perf_result FOREIGN KEY (result_id) REFERENCES AI_Results(result_id) ON DELETE CASCADE,
        CONSTRAINT fk_perf_image FOREIGN KEY (image_id) REFERENCES Images(image_id) ON DELETE CASCADE,
        CONSTRAINT fk_perf_model FOREIGN KEY (model_id) REFERENCES Models(model_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    
    # Performance Summary pro Inferenz
    """
    CREATE TABLE IF NOT EXISTS Performance_Summary (
        summary_id INT AUTO_INCREMENT PRIMARY KEY,
        result_id INT NOT NULL,
        image_id INT NOT NULL,
        model_id INT NOT NULL,
        
        -- Zeitbereich der Messung
        measurement_start DATETIME NOT NULL,
        measurement_end DATETIME NOT NULL,
        measurement_duration_ms INT NOT NULL,
        
        -- CPU Statistiken
        cpu_percent_avg FLOAT NULL,
        cpu_percent_max FLOAT NULL,
        cpu_percent_min FLOAT NULL,
        
        -- Memory Statistiken
        memory_used_avg BIGINT NULL,
        memory_used_max BIGINT NULL,
        memory_percent_avg FLOAT NULL,
        memory_percent_max FLOAT NULL,
        
        -- GPU Statistiken
        gpu_utilization_avg FLOAT NULL,
        gpu_utilization_max FLOAT NULL,
        gpu_memory_used_avg BIGINT NULL,
        gpu_memory_used_max BIGINT NULL,
        gpu_temperature_avg FLOAT NULL,
        gpu_temperature_max FLOAT NULL,
        
        -- I/O Statistiken
        disk_read_bytes_total BIGINT NULL,
        disk_write_bytes_total BIGINT NULL,
        network_bytes_total BIGINT NULL,
        
        -- Process Statistiken
        process_cpu_avg FLOAT NULL,
        process_cpu_max FLOAT NULL,
        process_memory_avg BIGINT NULL,
        process_memory_max BIGINT NULL,
        
        created_at DATETIME NOT NULL,
        
        CONSTRAINT fk_summ_result FOREIGN KEY (result_id) REFERENCES AI_Results(result_id) ON DELETE CASCADE,
        CONSTRAINT fk_summ_image FOREIGN KEY (image_id) REFERENCES Images(image_id) ON DELETE CASCADE,
        CONSTRAINT fk_summ_model FOREIGN KEY (model_id) REFERENCES Models(model_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
]

MONITORING_MIGRATIONS = [
    "CREATE INDEX ix_system_performance_timestamp ON System_Performance (timestamp)",
    "CREATE INDEX ix_system_performance_result_id ON System_Performance (result_id)",
    "CREATE INDEX ix_performance_summary_model_id ON Performance_Summary (model_id)",
    "CREATE INDEX ix_performance_summary_created_at ON Performance_Summary (created_at)"
]

# =========================
# System Monitoring Klassen
# =========================
class SystemMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics_history = []
        self.start_time = None
        self.disk_io_start = None
        self.network_io_start = None
        
    def get_current_metrics(self) -> Dict:
        """Sammelt aktuelle System-Metriken"""
        now = dt.datetime.now(tzlocal()).replace(tzinfo=None)
        metrics = {"timestamp": now}
        
        # CPU Metriken
        try:
            metrics.update({
                "cpu_percent": psutil.cpu_percent(interval=None),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
            })
            
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metrics.update({
                    "cpu_freq_current": cpu_freq.current,
                    "cpu_freq_max": cpu_freq.max
                })
        except Exception:
            pass
            
        # Memory Metriken
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            metrics.update({
                "memory_total": mem.total,
                "memory_available": mem.available,
                "memory_used": mem.used,
                "memory_percent": mem.percent,
                "memory_swap_total": swap.total,
                "memory_swap_used": swap.used,
            })
        except Exception:
            pass
            
        # GPU Metriken
        if MONITOR_GPU and GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Erste GPU
                    metrics.update({
                        "gpu_count": len(gpus),
                        "gpu_memory_total": int(gpu.memoryTotal * 1024 * 1024),  # MB -> Bytes
                        "gpu_memory_used": int(gpu.memoryUsed * 1024 * 1024),
                        "gpu_utilization": gpu.load * 100,
                        "gpu_temperature": gpu.temperature,
                    })
                    
                    # NVIDIA spezifische Metriken
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
                        metrics["gpu_power_draw"] = power
                    except Exception:
                        pass
            except Exception:
                pass
                
        # Disk I/O
        if MONITOR_DISK_IO:
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    metrics.update({
                        "disk_read_bytes": disk_io.read_bytes,
                        "disk_write_bytes": disk_io.write_bytes,
                        "disk_read_count": disk_io.read_count,
                        "disk_write_count": disk_io.write_count,
                    })
            except Exception:
                pass
                
        # Network I/O
        if MONITOR_NETWORK:
            try:
                net_io = psutil.net_io_counters()
                if net_io:
                    metrics.update({
                        "network_bytes_sent": net_io.bytes_sent,
                        "network_bytes_recv": net_io.bytes_recv,
                        "network_packets_sent": net_io.packets_sent,
                        "network_packets_recv": net_io.packets_recv,
                    })
            except Exception:
                pass
                
        # Process-spezifische Metriken
        try:
            with self.process.oneshot():
                metrics.update({
                    "process_cpu_percent": self.process.cpu_percent(),
                    "process_memory_rss": self.process.memory_info().rss,
                    "process_memory_vms": self.process.memory_info().vms,
                    "process_num_threads": self.process.num_threads(),
                })
                
                # File Descriptors (Unix only)
                try:
                    metrics["process_num_fds"] = self.process.num_fds()
                except (AttributeError, psutil.AccessDenied):
                    pass
        except Exception:
            pass
            
        return metrics
    
    def start_monitoring(self):
        """Startet kontinuierliches Monitoring"""
        self.monitoring = True
        self.metrics_history = []
        self.start_time = time.perf_counter()
        
        # Basis-Werte für Delta-Berechnungen
        if MONITOR_DISK_IO:
            self.disk_io_start = psutil.disk_io_counters()
        if MONITOR_NETWORK:
            self.network_io_start = psutil.net_io_counters()
            
        def monitor_loop():
            while self.monitoring:
                try:
                    metrics = self.get_current_metrics()
                    self.metrics_history.append(metrics)
                    time.sleep(SAMPLING_INTERVAL)
                except Exception:
                    pass
                    
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[Dict]:
        """Stoppt Monitoring und gibt gesammelte Metriken zurück"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        return self.metrics_history.copy()
    
    def get_summary_stats(self, metrics_history: List[Dict]) -> Dict:
        """Berechnet Summary-Statistiken aus Metriken-Historie"""
        if not metrics_history:
            return {}
            
        summary = {
            "measurement_start": metrics_history[0]["timestamp"],
            "measurement_end": metrics_history[-1]["timestamp"],
            "measurement_duration_ms": int((time.perf_counter() - self.start_time) * 1000),
        }
        
        # Numerische Felder für Statistiken
        numeric_fields = [
            "cpu_percent", "memory_used", "memory_percent",
            "gpu_utilization", "gpu_memory_used", "gpu_temperature",
            "process_cpu_percent", "process_memory_rss"
        ]
        
        for field in numeric_fields:
            values = [m.get(field) for m in metrics_history if m.get(field) is not None]
            if values:
                summary[f"{field}_avg"] = sum(values) / len(values)
                summary[f"{field}_max"] = max(values)
                if field in ["cpu_percent", "memory_percent", "gpu_utilization", "process_cpu_percent"]:
                    summary[f"{field}_min"] = min(values)
        
        # Delta-Berechnungen für I/O
        if len(metrics_history) >= 2:
            start_metrics = metrics_history[0]
            end_metrics = metrics_history[-1]
            
            # Disk I/O Delta
            if start_metrics.get("disk_read_bytes") and end_metrics.get("disk_read_bytes"):
                summary["disk_read_bytes_total"] = end_metrics["disk_read_bytes"] - start_metrics["disk_read_bytes"]
                summary["disk_write_bytes_total"] = end_metrics["disk_write_bytes"] - start_metrics["disk_write_bytes"]
            
            # Network I/O Delta
            if MONITOR_NETWORK and start_metrics.get("network_bytes_sent") and end_metrics.get("network_bytes_sent"):
                sent_delta = end_metrics["network_bytes_sent"] - start_metrics["network_bytes_sent"]
                recv_delta = end_metrics["network_bytes_recv"] - start_metrics["network_bytes_recv"]
                summary["network_bytes_total"] = sent_delta + recv_delta
                
        return summary

# =========================
# DB Integration
# =========================
def ensure_monitoring_schema(conn):
    """Erweitert das DB-Schema um Monitoring-Tabellen"""
    cur = conn.cursor()
    for sql in SYSTEM_MONITORING_TABLES:
        cur.execute(sql)
    
    # Migrations tolerant ausführen
    for sql in MONITORING_MIGRATIONS:
        try:
            cur.execute(sql)
        except mysql.connector.Error:
            pass
    conn.commit()
    cur.close()

def insert_performance_metrics(conn, result_id: int, image_id: int, model_id: int, 
                              metrics_history: List[Dict]):
    """Speichert detaillierte Performance-Metriken"""
    if not metrics_history:
        return
        
    cur = conn.cursor()
    
    # Einzelne Metriken-Punkte einfügen (Sample-reduziert für DB-Effizienz)
    step = max(1, len(metrics_history) // 100)  # Max 100 Datenpunkte pro Inferenz
    
    for metrics in metrics_history[::step]:
        cur.execute("""
            INSERT INTO System_Performance (
                result_id, image_id, model_id, timestamp,
                cpu_percent, cpu_count_logical, cpu_count_physical, 
                cpu_freq_current, cpu_freq_max,
                memory_total, memory_available, memory_used, memory_percent,
                memory_swap_total, memory_swap_used,
                gpu_count, gpu_memory_total, gpu_memory_used, gpu_utilization,
                gpu_temperature, gpu_power_draw,
                disk_read_bytes, disk_write_bytes, disk_read_count, disk_write_count,
                network_bytes_sent, network_bytes_recv, network_packets_sent, network_packets_recv,
                process_cpu_percent, process_memory_rss, process_memory_vms,
                process_num_threads, process_num_fds
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                      %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            result_id, image_id, model_id, metrics["timestamp"],
            metrics.get("cpu_percent"), metrics.get("cpu_count_logical"), metrics.get("cpu_count_physical"),
            metrics.get("cpu_freq_current"), metrics.get("cpu_freq_max"),
            metrics.get("memory_total"), metrics.get("memory_available"), 
            metrics.get("memory_used"), metrics.get("memory_percent"),
            metrics.get("memory_swap_total"), metrics.get("memory_swap_used"),
            metrics.get("gpu_count"), metrics.get("gpu_memory_total"), 
            metrics.get("gpu_memory_used"), metrics.get("gpu_utilization"),
            metrics.get("gpu_temperature"), metrics.get("gpu_power_draw"),
            metrics.get("disk_read_bytes"), metrics.get("disk_write_bytes"),
            metrics.get("disk_read_count"), metrics.get("disk_write_count"),
            metrics.get("network_bytes_sent"), metrics.get("network_bytes_recv"),
            metrics.get("network_packets_sent"), metrics.get("network_packets_recv"),
            metrics.get("process_cpu_percent"), metrics.get("process_memory_rss"),
            metrics.get("process_memory_vms"), metrics.get("process_num_threads"),
            metrics.get("process_num_fds")
        ))
    
    conn.commit()
    cur.close()

def insert_performance_summary(conn, result_id: int, image_id: int, model_id: int, 
                              summary_stats: Dict):
    """Speichert Performance-Summary"""
    if not summary_stats:
        return
        
    now = dt.datetime.now(tzlocal()).replace(tzinfo=None)
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO Performance_Summary (
            result_id, image_id, model_id,
            measurement_start, measurement_end, measurement_duration_ms,
            cpu_percent_avg, cpu_percent_max, cpu_percent_min,
            memory_used_avg, memory_used_max, memory_percent_avg, memory_percent_max,
            gpu_utilization_avg, gpu_utilization_max, 
            gpu_memory_used_avg, gpu_memory_used_max,
            gpu_temperature_avg, gpu_temperature_max,
            disk_read_bytes_total, disk_write_bytes_total, network_bytes_total,
            process_cpu_avg, process_cpu_max, process_memory_avg, process_memory_max,
            created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        result_id, image_id, model_id,
        summary_stats.get("measurement_start"), summary_stats.get("measurement_end"),
        summary_stats.get("measurement_duration_ms"),
        summary_stats.get("cpu_percent_avg"), summary_stats.get("cpu_percent_max"), summary_stats.get("cpu_percent_min"),
        summary_stats.get("memory_used_avg"), summary_stats.get("memory_used_max"),
        summary_stats.get("memory_percent_avg"), summary_stats.get("memory_percent_max"),
        summary_stats.get("gpu_utilization_avg"), summary_stats.get("gpu_utilization_max"),
        summary_stats.get("gpu_memory_used_avg"), summary_stats.get("gpu_memory_used_max"),
        summary_stats.get("gpu_temperature_avg"), summary_stats.get("gpu_temperature_max"),
        summary_stats.get("disk_read_bytes_total"), summary_stats.get("disk_write_bytes_total"),
        summary_stats.get("network_bytes_total"),
        summary_stats.get("process_cpu_percent_avg"), summary_stats.get("process_cpu_percent_max"),
        summary_stats.get("process_memory_rss_avg"), summary_stats.get("process_memory_rss_max"),
        now
    ))
    
    conn.commit()
    cur.close()

# =========================
# Context Manager für einfache Nutzung
# =========================
class ResourceMonitorContext:
    """Context Manager für einfaches Ressourcen-Monitoring"""
    
    def __init__(self, conn, image_id: int, model_id: int):
        self.conn = conn
        self.image_id = image_id
        self.model_id = model_id
        self.monitor = SystemMonitor()
        self.result_id = None
        
    def __enter__(self):
        self.monitor.start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        metrics_history = self.monitor.stop_monitoring()
        
        if self.result_id and metrics_history:
            try:
                # Detaillierte Metriken speichern
                insert_performance_metrics(self.conn, self.result_id, self.image_id, 
                                         self.model_id, metrics_history)
                
                # Summary-Statistiken speichern
                summary_stats = self.monitor.get_summary_stats(metrics_history)
                insert_performance_summary(self.conn, self.result_id, self.image_id,
                                         self.model_id, summary_stats)
                
                print(f"[PERF] Saved {len(metrics_history)} metrics, "
                      f"CPU avg: {summary_stats.get('cpu_percent_avg', 0):.1f}%, "
                      f"Memory avg: {summary_stats.get('memory_percent_avg', 0):.1f}%"
                      + (f", GPU avg: {summary_stats.get('gpu_utilization_avg', 0):.1f}%" if MONITOR_GPU else ""))
                      
            except Exception as e:
                print(f"[WARNING] Failed to save performance metrics: {e}")
    
    def set_result_id(self, result_id: int):
        """Setzt die result_id nach dem Einfügen des AI_Results"""
        self.result_id = result_id