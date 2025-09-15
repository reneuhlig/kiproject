#!/bin/bash
# setup_ollama.sh - Installiert und konfiguriert Ollama mit Gemma 3

# =============================================================================
# OLLAMA INSTALLATION UND SETUP
# =============================================================================

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

install_ollama() {
    log_message "Installiere Ollama..."
    
    # Prüfe ob bereits installiert
    if command -v ollama &> /dev/null; then
        log_message "✓ Ollama bereits installiert: $(ollama --version)"
        return 0
    fi
    
    # Installiere Ollama
    curl -fsSL https://ollama.com/install.sh | sh
    
    if command -v ollama &> /dev/null; then
        log_message "✓ Ollama erfolgreich installiert"
        return 0
    else
        log_message "✗ Ollama Installation fehlgeschlagen"
        return 1
    fi
}

start_ollama_service() {
    log_message "Starte Ollama Service..."
    
    # Starte Ollama im Hintergrund
    ollama serve &
    OLLAMA_PID=$!
    
    # Warte bis Service bereit ist
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log_message "✓ Ollama Service läuft (PID: $OLLAMA_PID)"
            return 0
        fi
        sleep 2
    done
    
    log_message "✗ Ollama Service konnte nicht gestartet werden"
    return 1
}

pull_gemma3_models() {
    log_message "Lade Gemma 3 Modelle herunter..."
    
    # Verfügbare Modelle (nach Größe sortiert)
    models=("gemma3:270m" "gemma3:1b" "gemma3:4b")
    
    for model in "${models[@]}"; do
        log_message "Lade $model herunter..."
        
        # Model herunterladen
        ollama pull "$model"
        
        if [ $? -eq 0 ]; then
            log_message "✓ $model erfolgreich heruntergeladen"
        else
            log_message "✗ Fehler beim Herunterladen von $model"
        fi
        
        # Kurze Pause zwischen Downloads
        sleep 2
    done
    
    # Liste verfügbare Modelle
    log_message "Verfügbare Modelle:"
    ollama list
}

test_gemma3_vision() {
    log_message "Teste Gemma 3 Vision-Funktionalität..."
    
    # Erstelle Testbild
    python3 -c "
import numpy as np
import cv2
import base64

# Erstelle einfaches Testbild mit Person
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.circle(img, (200, 120), 30, (255, 200, 150), -1)  # Kopf
cv2.rectangle(img, (170, 150), (230, 280), (100, 100, 255), -1)  # Körper

# Speichere Testbild
cv2.imwrite('test_person.jpg', img)
print('✓ Testbild erstellt: test_person.jpg')
"
    
    if [ ! -f "test_person.jpg" ]; then
        log_message "✗ Testbild konnte nicht erstellt werden"
        return 1
    fi
    
    # Teste mit curl und base64
    log_message "Teste Gemma 3 Vision mit curl..."
    
    # Base64 kodieren
    image_base64=$(base64 -w 0 test_person.jpg)
    
    # Test mit gemma3:270m (kleinste/schnellste Version)
    response=$(curl -s http://localhost:11434/api/generate -d '{
        "model": "gemma3:270m",
        "prompt": "How many people can you see in this image? Give me only a number.",
        "images": ["'$image_base64'"],
        "stream": false
    }')
    
    if [ $? -eq 0 ]; then
        echo "API Response:"
        echo "$response" | jq -r '.response' 2>/dev/null || echo "$response"
        log_message "✓ Gemma 3 Vision Test erfolgreich"
    else
        log_message "✗ Gemma 3 Vision Test fehlgeschlagen"
        return 1
    fi
    
    # Aufräumen
    rm -f test_person.jpg
}

test_ollama_python() {
    log_message "Teste Ollama Python Library..."
    
    # Installiere ollama-python falls nicht vorhanden
    pip install ollama 2>/dev/null || pip3 install ollama
    
    # Python-Test
    python3 -c "
import ollama
import base64
import numpy as np
import cv2

try:
    # Erstelle Testbild
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 80), 20, (255, 200, 150), -1)  # Kopf
    cv2.rectangle(img, (80, 100), (120, 160), (100, 100, 255), -1)  # Körper
    cv2.imwrite('test_ollama.jpg', img)
    
    # Base64 kodieren
    with open('test_ollama.jpg', 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Ollama Python API verwenden
    response = ollama.generate(
        model='gemma3:270m',
        prompt='Count the people in this image. Answer with only a number.',
        images=[image_data]
    )
    
    print('✓ Ollama Python Test erfolgreich')
    print('Response:', response['response'].strip())
    
    # Aufräumen
    import os
    os.remove('test_ollama.jpg')
    
except Exception as e:
    print(f'✗ Ollama Python Test fehlgeschlagen: {e}')
"
}

create_ollama_config() {
    log_message "Erstelle Ollama-Konfiguration..."
    
    # Systemd Service für automatischen Start (falls systemd verfügbar)
    if command -v systemctl &> /dev/null; then
        sudo tee /etc/systemd/system/ollama.service > /dev/null << EOF
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=ollama
Group=ollama
ExecStart=/usr/local/bin/ollama serve
Environment=OLLAMA_HOST=0.0.0.0:11434
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
        
        # Erstelle ollama Benutzer
        sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama 2>/dev/null || true
        
        # Service aktivieren
        sudo systemctl daemon-reload
        sudo systemctl enable ollama
        sudo systemctl start ollama
        
        log_message "✓ Ollama Systemd Service konfiguriert"
    fi
    
    # Umgebungsvariablen setzen
    echo 'export OLLAMA_HOST=localhost:11434' >> ~/.bashrc
    
    log_message "✓ Ollama-Konfiguration erstellt"
}

show_usage_examples() {
    log_message "Verwendungsbeispiele:"
    
    echo ""
    echo "=== CURL BEISPIELE ==="
    echo "# Einfache Textanfrage:"
    echo "curl http://localhost:11434/api/generate -d '{\"model\":\"gemma3:270m\",\"prompt\":\"Hello\"}'"
    echo ""
    echo "# Vision-Anfrage (mit base64-kodiertem Bild):"
    echo "curl http://localhost:11434/api/generate -d '{\"model\":\"gemma3:4b\",\"prompt\":\"Describe this image\",\"images\":[\"<base64_image>\"]}'"
    echo ""
    
    echo "=== PYTHON BEISPIELE ==="
    echo "import ollama"
    echo "# Textgeneration:"
    echo "response = ollama.generate(model='gemma3:270m', prompt='Hello')"
    echo ""
    echo "# Vision:"
    echo "with open('image.jpg', 'rb') as f:"
    echo "    image_data = base64.b64encode(f.read()).decode()"
    echo "response = ollama.generate(model='gemma3:4b', prompt='Count people', images=[image_data])"
    echo ""
    
    echo "=== VERFÜGBARE MODELLE ==="
    echo "gemma3:270m  - 292MB  - Kleinste/schnellste Version"
    echo "gemma3:1b    - 1.4GB  - Gute Balance"
    echo "gemma3:4b    - 3.3GB  - Beste Qualität für die meisten Anwendungen"
    echo "gemma3:12b   - 8.8GB  - Hohe Qualität (viel RAM nötig)"
    echo "gemma3:27b   - 19GB   - Beste Qualität (sehr viel RAM nötig)"
}

# =============================================================================
# HAUPTLOGIK
# =============================================================================

case "${1:-install}" in
    "install")
        log_message "=== VOLLSTÄNDIGE OLLAMA INSTALLATION ==="
        install_ollama
        start_ollama_service
        pull_gemma3_models
        test_gemma3_vision
        test_ollama_python
        create_ollama_config
        show_usage_examples
        ;;
    "start")
        start_ollama_service
        ;;
    "pull")
        pull_gemma3_models
        ;;
    "test")
        test_gemma3_vision
        test_ollama_python
        ;;
    "config")
        create_ollama_config
        ;;
    "examples")
        show_usage_examples
        ;;
    "status")
        log_message "Ollama Status:"
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log_message "✓ Ollama Service läuft"
            ollama list
        else
            log_message "✗ Ollama Service läuft nicht"
        fi
        ;;
    "help"|*)
        echo "Usage: $0 {install|start|pull|test|config|examples|status}"
        echo ""
        echo "Commands:"
        echo "  install   - Vollständige Installation und Setup"
        echo "  start     - Starte nur den Ollama Service"
        echo "  pull      - Lade Gemma 3 Modelle herunter"
        echo "  test      - Teste Vision-Funktionalität"
        echo "  config    - Konfiguriere Systemd Service"
        echo "  examples  - Zeige Verwendungsbeispiele"
        echo "  status    - Zeige Service-Status und verfügbare Modelle"
        ;;
esac