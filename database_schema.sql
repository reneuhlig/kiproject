-- Datenbankschema für AI-Personenerkennung
-- Erstellt Tabellen für Runs und Ergebnisse aller KI-Modelle

CREATE DATABASE IF NOT EXISTS ai_detection CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE ai_detection;

-- Tabelle für Run-Informationen (erweitert für alle Modelle)
CREATE TABLE IF NOT EXISTS ai_runs (
    run_id VARCHAR(36) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    total_images INT DEFAULT 0,
    successful_detections INT DEFAULT 0,
    failed_detections INT DEFAULT 0,
    avg_processing_time FLOAT,
    total_processing_time FLOAT,
    avg_cpu_usage FLOAT,
    max_cpu_usage FLOAT,
    avg_memory_usage FLOAT,
    max_memory_usage FLOAT,
    avg_gpu_usage FLOAT,
    max_gpu_usage FLOAT,
    status ENUM('running', 'completed', 'failed', 'cancelled') DEFAULT 'running',
    error_message TEXT,
    config_json TEXT,
    INDEX idx_model_name (model_name),
    INDEX idx_start_time (start_time),
    INDEX idx_status (status)
);

-- Tabelle für Erkennungsergebnisse (generisch für alle Modelle)
CREATE TABLE IF NOT EXISTS detection_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(36) NOT NULL,
    image_path VARCHAR(500) NOT NULL,
    image_filename VARCHAR(255) NOT NULL,
    classification VARCHAR(100),
    model_output JSON,
    confidence_scores TEXT,
    processing_time FLOAT NOT NULL,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Zusätzliche Felder für Personenerkennung
    persons_detected INT DEFAULT 0,
    avg_confidence FLOAT,
    max_confidence FLOAT,
    min_confidence FLOAT,
    is_uncertain BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (run_id) REFERENCES ai_runs(run_id) ON DELETE CASCADE,
    INDEX idx_run_id (run_id),
    INDEX idx_classification (classification),
    INDEX idx_timestamp (timestamp),
    INDEX idx_persons_detected (persons_detected),
    INDEX idx_success (success)
);

-- View für Run-Übersicht
CREATE OR REPLACE VIEW run_overview AS
SELECT 
    r.run_id,
    r.model_name,
    r.model_version,
    r.start_time,
    r.end_time,
    r.total_images,
    r.successful_detections,
    r.failed_detections,
    r.status,
    ROUND(r.avg_processing_time, 3) as avg_processing_time,
    ROUND(r.total_processing_time, 1) as total_processing_time,
    ROUND(r.avg_cpu_usage, 1) as avg_cpu_usage,
    ROUND(r.avg_memory_usage, 1) as avg_memory_usage,
    ROUND(r.avg_gpu_usage, 1) as avg_gpu_usage,
    -- Zusätzliche Statistiken aus detection_results
    ROUND(AVG(dr.persons_detected), 2) as avg_persons_per_image,
    SUM(dr.persons_detected) as total_persons_detected,
    COUNT(CASE WHEN dr.is_uncertain = TRUE THEN 1 END) as uncertain_detections,
    ROUND(AVG(dr.avg_confidence), 3) as overall_avg_confidence
FROM ai_runs r
LEFT JOIN detection_results dr ON r.run_id = dr.run_id
GROUP BY r.run_id, r.model_name, r.model_version, r.start_time, r.end_time,
         r.total_images, r.successful_detections, r.failed_detections, r.status,
         r.avg_processing_time, r.total_processing_time, r.avg_cpu_usage,
         r.avg_memory_usage, r.avg_gpu_usage;

-- View für Klassifizierungsanalyse
CREATE OR REPLACE VIEW classification_analysis AS
SELECT 
    classification,
    COUNT(*) as total_images,
    AVG(persons_detected) as avg_persons,
    SUM(persons_detected) as total_persons,
    COUNT(CASE WHEN persons_detected > 0 THEN 1 END) as images_with_people,
    COUNT(CASE WHEN persons_detected = 0 THEN 1 END) as images_without_people,
    COUNT(CASE WHEN is_uncertain = TRUE THEN 1 END) as uncertain_results,
    ROUND(AVG(avg_confidence), 3) as avg_confidence,
    COUNT(CASE WHEN success = FALSE THEN 1 END) as failed_detections
FROM detection_results
WHERE success = TRUE
GROUP BY classification
ORDER BY total_images DESC;

-- View für Modellvergleich
CREATE OR REPLACE VIEW model_comparison AS
SELECT 
    model_name,
    model_version,
    COUNT(DISTINCT run_id) as total_runs,
    SUM(total_images) as total_images_processed,
    SUM(successful_detections) as total_successful,
    SUM(failed_detections) as total_failed,
    ROUND(AVG(avg_processing_time), 3) as avg_time_per_image,
    ROUND(AVG(avg_cpu_usage), 1) as avg_cpu,
    ROUND(AVG(avg_memory_usage), 1) as avg_memory,
    ROUND(AVG(avg_gpu_usage), 1) as avg_gpu,
    -- Performance-Metriken
    ROUND(SUM(successful_detections) / NULLIF(SUM(total_images), 0) * 100, 2) as success_rate_percent
FROM ai_runs
WHERE status = 'completed'
GROUP BY model_name, model_version
ORDER BY total_images_processed DESC;

-- Trigger für automatische Aktualisierung der Personenerkennungsfelder
DELIMITER //
CREATE TRIGGER update_person_detection_fields
    BEFORE INSERT ON detection_results
    FOR EACH ROW
BEGIN
    -- Extrahiere Personenanzahl aus model_output JSON
    IF NEW.model_output IS NOT NULL THEN
        SET NEW.persons_detected = COALESCE(JSON_EXTRACT(NEW.model_output, '$.persons_detected'), 0);
        SET NEW.avg_confidence = COALESCE(JSON_EXTRACT(NEW.model_output, '$.avg_confidence'), 0.0);
        SET NEW.max_confidence = COALESCE(JSON_EXTRACT(NEW.model_output, '$.max_confidence'), 0.0);  
        SET NEW.min_confidence = COALESCE(JSON_EXTRACT(NEW.model_output, '$.min_confidence'), 0.0);
        SET NEW.is_uncertain = COALESCE(JSON_EXTRACT(NEW.model_output, '$.uncertain'), FALSE);
    END IF;
END//
DELIMITER ;

-- Beispielabfragen (als Kommentare für Referenz)

/*
-- Top 10 Bilder mit den meisten Personen
SELECT image_filename, classification, persons_detected, avg_confidence, run_id
FROM detection_results 
WHERE success = TRUE 
ORDER BY persons_detected DESC, avg_confidence DESC 
LIMIT 10;

-- Durchschnittliche Erkennungszeit pro Modell
SELECT model_name, 
       COUNT(*) as runs, 
       ROUND(AVG(avg_processing_time), 3) as avg_time_per_image,
       ROUND(AVG(total_processing_time), 1) as avg_total_time
FROM ai_runs 
WHERE status = 'completed' 
GROUP BY model_name;

-- Unsichere Erkennungen pro Klassifizierung
SELECT classification,
       COUNT(*) as total,
       COUNT(CASE WHEN is_uncertain = TRUE THEN 1 END) as uncertain,
       ROUND(COUNT(CASE WHEN is_uncertain = TRUE THEN 1 END) / COUNT(*) * 100, 2) as uncertain_percentage
FROM detection_results 
WHERE success = TRUE
GROUP BY classification
ORDER BY uncertain_percentage DESC;

-- Performance-Analyse nach Systemressourcen
SELECT model_name,
       CASE 
           WHEN avg_cpu_usage < 50 THEN 'Low CPU'
           WHEN avg_cpu_usage < 80 THEN 'Medium CPU'  
           ELSE 'High CPU'
       END as cpu_usage_category,
       COUNT(*) as runs,
       ROUND(AVG(avg_processing_time), 3) as avg_time,
       ROUND(AVG(successful_detections / NULLIF(total_images, 0) * 100), 2) as success_rate
FROM ai_runs 
WHERE status = 'completed'
GROUP BY model_name, cpu_usage_category
ORDER BY model_name, cpu_usage_category;
*/

-- Benutzer und Berechtigungen (optional)
-- CREATE USER 'ai_detection_user'@'localhost' IDENTIFIED BY 'secure_password_here';
-- GRANT SELECT, INSERT, UPDATE ON ai_detection.* TO 'ai_detection_user'@'localhost';
-- FLUSH PRIVILEGES;