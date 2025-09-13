import requests
import base64
import json
import re
from typing import Dict, Any, List
from PIL import Image
from BaseDetector import BaseDetector


class OllamaGemma3PersonDetector(BaseDetector):
    """Ollama Gemma 3 Detektor für Personenerkennung über Vision-Language Model"""
    
    def __init__(self, model_name: str = "gemma3:4b", confidence_threshold: float = 0.5, 
                 ollama_host: str = "http://localhost:11434"):
        """
        Initialisiert den Ollama Gemma 3 Detektor
        
        Args:
            model_name: Ollama Modellname (gemma3:270m, gemma3:1b, gemma3:4b, gemma3:12b, gemma3:27b)
            confidence_threshold: Mindest-Konfidenz für unsichere Antworten
            ollama_host: Ollama Server URL
        """
        super().__init__("Ollama-Gemma3", model_name.split(':')[-1])
        self.confidence_threshold = confidence_threshold
        self.model_name_ollama = model_name
        self.ollama_host = ollama_host
        self.api_url = f"{ollama_host}/api/generate"
        
        # Teste Verbindung und prüfe ob Modell verfügbar ist
        self._check_ollama_connection()
        self._ensure_model_available()
        
    def _check_ollama_connection(self):
        """Prüft Ollama-Verbindung"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                print(f"✓ Ollama-Verbindung erfolgreich: {self.ollama_host}")
            else:
                raise Exception(f"Ollama nicht erreichbar: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama-Verbindung fehlgeschlagen: {e}")
    
    def _ensure_model_available(self):
        """Prüft ob Modell verfügbar ist, lädt es ggf. herunter"""
        try:
            # Liste verfügbare Modelle
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                if self.model_name_ollama in available_models:
                    print(f"✓ Modell {self.model_name_ollama} ist verfügbar")
                    return
            
            # Modell nicht gefunden - versuche herunterzuladen
            print(f"Modell {self.model_name_ollama} nicht gefunden. Lade herunter...")
            self._pull_model()
            
        except Exception as e:
            print(f"⚠ Warnung: Modell-Verfügbarkeit konnte nicht geprüft werden: {e}")
    
    def _pull_model(self):
        """Lädt Modell herunter"""
        pull_data = {"name": self.model_name_ollama}
        
        try:
            response = requests.post(f"{self.ollama_host}/api/pull", 
                                   json=pull_data, 
                                   timeout=300)  # 5 Minuten Timeout
            
            if response.status_code == 200:
                print(f"✓ Modell {self.model_name_ollama} erfolgreich heruntergeladen")
            else:
                print(f"⚠ Modell-Download möglicherweise fehlgeschlagen: HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("⚠ Modell-Download dauert länger - läuft im Hintergrund weiter")
        except Exception as e:
            print(f"⚠ Modell-Download-Fehler: {e}")
    
    def _encode_image(self, image_path: str) -> str:
        """Kodiert Bild als base64"""
        try:
            # Bild laden und optimieren
            with Image.open(image_path) as img:
                # RGB konvertieren falls nötig
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Größe reduzieren falls sehr groß (für Performance)
                if img.size[0] > 1024 or img.size[1] > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                # Temporär speichern und base64 kodieren
                import io
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
                
        except Exception as e:
            raise Exception(f"Bild-Kodierung fehlgeschlagen: {e}")
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Erkennt Personen über Ollama Gemma 3 Vision
        
        Args:
            image_path: Pfad zum Bild
            
        Returns:
            Dictionary mit Erkennungsergebnissen
        """
        try:
            # Bild kodieren
            image_base64 = self._encode_image(image_path)
            
            # Prompt für Personenzählung
            prompt = """Look at this image carefully and count the number of people you can see. 
Give me ONLY a number as your answer - nothing else. If you see no people, answer 0."""
            
            # API-Request vorbereiten
            data = {
                "model": self.model_name_ollama,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 10  # Kurze Antwort erwarten
                }
            }
            
            # Request senden
            response = requests.post(self.api_url, json=data, timeout=120)
            
            if response.status_code != 200:
                raise Exception(f"Ollama API Fehler: HTTP {response.status_code}")
            
            result_data = response.json()
            raw_response = result_data.get('response', '').strip()
            
            # Parse Antwort
            return self._parse_response(raw_response, image_path)
            
        except Exception as e:
            print(f"Fehler in Ollama Gemma 3 Detection: {e}")
            return self._create_error_result(str(e))
    
    def _parse_response(self, response: str, image_path: str) -> Dict[str, Any]:
        """Parst die Modellantwort"""
        try:
            # Extrahiere Nummer aus Antwort
            numbers = re.findall(r'\b(\d+)\b', response)
            if numbers:
                person_count = int(numbers[0])
                # Begrenze auf sinnvollen Bereich
                person_count = max(0, min(person_count, 50))
            else:
                person_count = 0
            
            # Schätze Konfidenz basierend auf Antwort
            confidence = self._estimate_confidence(response, person_count)
            
            # Erstelle Person-Objekte
            persons = []
            confidences = []
            
            if person_count > 0:
                base_confidence = max(0.4, confidence)
                for i in range(person_count):
                    person_confidence = max(0.2, base_confidence * (0.95 ** i))
                    persons.append({
                        'bbox': [0, 0, 100, 100],  # Placeholder bbox
                        'confidence': person_confidence,
                        'class': 'person',
                        'detection_method': 'ollama_gemma3_vision'
                    })
                    confidences.append(person_confidence)
            
            avg_conf = confidence if confidences else 0.0
            max_conf = max(confidences) if confidences else 0.0
            min_conf = min(confidences) if confidences else 0.0
            
            return {
                'persons_detected': person_count,
                'persons': persons,
                'confidences': confidences,
                'avg_confidence': float(avg_conf),
                'max_confidence': float(max_conf),
                'min_confidence': float(min_conf),
                'uncertain': confidence < 0.7 or any(word in response.lower() for word in ['unsure', 'maybe', 'difficult']),
                'model_output': {
                    'raw_response': response,
                    'ollama_model': self.model_name_ollama,
                    'estimated_confidence': float(confidence)
                }
            }
            
        except Exception as e:
            print(f"Parse-Fehler: {e}")
            return self._create_error_result(f"Parse error: {e}", response)
    
    def _estimate_confidence(self, response: str, person_count: int) -> float:
        """Schätzt Konfidenz basierend auf Antwort"""
        if not response:
            return 0.3
        
        response_lower = response.lower()
        
        # Wenn nur eine Zahl (gute klare Antwort)
        if re.match(r'^\s*\d+\s*$', response.strip()):
            return 0.85
        
        # Negative Indikatoren
        if any(word in response_lower for word in ['unsure', 'difficult', 'hard to tell', 'maybe', 'possibly']):
            return 0.4
        
        # Positive Indikatoren
        if any(word in response_lower for word in ['clearly', 'obviously', 'definitely', 'certain']):
            return 0.9
        
        # Längere Erklärung meist weniger sicher
        if len(response) > 50:
            return 0.6
        
        # Standard für kurze, direkte Antworten
        return 0.7
    
    def _create_error_result(self, error_msg: str, response: str = "") -> Dict[str, Any]:
        """Erstellt einheitliches Fehler-Ergebnis"""
        return {
            'persons_detected': 0,
            'persons': [],
            'confidences': [],
            'avg_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 0.0,
            'uncertain': True,
            'error': error_msg,
            'model_output': {
                'raw_response': response,
                'error_details': error_msg,
                'ollama_model': self.model_name_ollama
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Modellinformationen zurück"""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'framework': 'Ollama/Gemma3',
            'ollama_model': self.model_name_ollama,
            'ollama_host': self.ollama_host,
            'confidence_threshold': self.confidence_threshold,
            'task': 'person_detection_via_ollama_vision',
            'input_size': 'variable',
            'classes': ['person']
        }