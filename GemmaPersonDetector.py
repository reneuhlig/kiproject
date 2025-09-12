import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import re
import json
from typing import Dict, Any, List
from BaseDetector import BaseDetector


class GemmaPersonDetector(BaseDetector):
    """Gemma/LLaVA Detektor für Personenerkennung über Vision-Language Model"""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", confidence_threshold: float = 0.5):
        """
        Initialisiert den Gemma/LLaVA Detektor
        
        Args:
            model_name: HuggingFace Modellname
            confidence_threshold: Mindest-Konfidenz für unsichere Antworten
        """
        super().__init__("Gemma/LLaVA", "1.5-7b")
        self.confidence_threshold = confidence_threshold
        self.model_name_hf = model_name
        
        # Modell und Processor laden
        try:
            print(f"Lade {model_name} - das kann einige Minuten dauern...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            # Gerätezuordnung optimieren
            device_map = "auto" if torch.cuda.is_available() else None
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map=device_map
            )
            
            if not torch.cuda.is_available():
                print("⚠ Kein CUDA verfügbar - verwende CPU (sehr langsam)")
            else:
                print(f"✓ Verwende GPU: {torch.cuda.get_device_name(0)}")
                
            print(f"✓ {model_name} erfolgreich geladen")
        except Exception as e:
            print(f"✗ Fehler beim Laden des Modells: {e}")
            raise
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Erkennt Personen über Vision-Language Understanding
        
        Args:
            image_path: Pfad zum Bild
            
        Returns:
            Dictionary mit Erkennungsergebnissen
        """
        try:
            # Bild laden und Größe prüfen
            image = Image.open(image_path).convert('RGB')
            
            # Bild verkleinern falls sehr groß (für Performance)
            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Prompt für Personenzählung
            prompt = "USER: <image>\nCount the number of people in this image. Give me only a number.\nASSISTANT:"
            
            # Input vorbereiten
            inputs = self.processor(prompt, image, return_tensors="pt")
            
            # Inputs auf das richtige Gerät verschieben
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generation mit Timeout-Schutz
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,  # Reduziert für einfache Antwort
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
                    )
                except Exception as gen_error:
                    print(f"Generation-Fehler: {gen_error}")
                    return self._create_error_result(f"Generation failed: {gen_error}")
            
            # Response dekodieren
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extrahiere nur den Assistant-Teil
            if "ASSISTANT:" in response:
                assistant_response = response.split("ASSISTANT:")[-1].strip()
            else:
                assistant_response = response.strip()
            
            # Parse die Antwort
            result = self._parse_response(assistant_response, image_path)
            
            return result
            
        except Exception as e:
            print(f"Fehler in Gemma Detection: {e}")
            return self._create_error_result(str(e))
    
    def _parse_response(self, response: str, image_path: str) -> Dict[str, Any]:
        """Parst die Modellantwort"""
        try:
            # Versuche JSON zu extrahieren
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group()
                    parsed = json.loads(json_str)
                    
                    person_count = parsed.get('person_count', 0)
                    confidence = parsed.get('confidence', 0.5)
                    description = parsed.get('description', '')
                    uncertain_areas = parsed.get('uncertain_areas', '')
                except json.JSONDecodeError:
                    # JSON parsing fehlgeschlagen, verwende Fallback
                    person_count, confidence, description, uncertain_areas = self._fallback_parse(response)
            else:
                # Kein JSON gefunden, verwende Fallback
                person_count, confidence, description, uncertain_areas = self._fallback_parse(response)
            
            # Validierung
            person_count = max(0, int(person_count))
            confidence = max(0.0, min(1.0, float(confidence)))
            
            # Erstelle einheitliche Person-Objekte
            persons = []
            confidences = []
            
            if person_count > 0:
                # Da wir keine genauen Bounding Boxes haben, erstelle generische Einträge
                base_confidence = max(0.3, confidence)
                for i in range(person_count):
                    # Leicht reduzierte Konfidenz für weitere Personen
                    person_confidence = max(0.1, base_confidence * (0.95 ** i))
                    persons.append({
                        'bbox': [0, 0, 100, 100],  # Placeholder bbox
                        'confidence': person_confidence,
                        'class': 'person',
                        'detection_method': 'llm_analysis'
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
                'uncertain': confidence < 0.7 or len(str(uncertain_areas).strip()) > 0,
                'model_output': {
                    'raw_response': response,
                    'description': str(description),
                    'uncertain_areas': str(uncertain_areas),
                    'llm_confidence': float(confidence)
                }
            }
            
        except Exception as e:
            print(f"Parse-Fehler: {e}")
            return self._create_error_result(f"Parse error: {e}", response)
    
    def _fallback_parse(self, response: str) -> tuple:
        """Fallback-Parsing wenn kein JSON vorhanden"""
        try:
            # Suche nach Zahlen im Text
            numbers = re.findall(r'\b(\d+)\b', response)
            person_count = int(numbers[0]) if numbers else 0
            
            # Begrenze auf sinnvollen Bereich
            person_count = min(person_count, 20)  # Max 20 Personen
            
            # Schätze Konfidenz basierend auf Schlüsselwörtern
            confidence = self._estimate_confidence_from_text(response)
            
            description = response[:100] + "..." if len(response) > 100 else response
            uncertain_areas = "Parsed from unstructured response"
            
            return person_count, confidence, description, uncertain_areas
            
        except Exception as e:
            print(f"Fallback-Parse-Fehler: {e}")
            return 0, 0.3, response, "Parse error in fallback"
    
    def _estimate_confidence_from_text(self, text: str) -> float:
        """Schätzt Konfidenz basierend auf Textinhalt"""
        if not text:
            return 0.3
            
        text_lower = text.lower()
        
        # Hohe Konfidenz Indikatoren
        high_conf_words = ['clearly', 'obviously', 'definitely', 'certainly', 'exactly', 'precisely']
        if any(word in text_lower for word in high_conf_words):
            return 0.9
        
        # Niedrige Konfidenz Indikatoren
        low_conf_words = ['maybe', 'possibly', 'might', 'unclear', 'difficult', 'hard to', 'unsure', 'perhaps']
        if any(word in text_lower for word in low_conf_words):
            return 0.4
        
        # Mittlere Konfidenz Indikatoren
        med_conf_words = ['appears', 'seems', 'likely', 'probably', 'looks like']
        if any(word in text_lower for word in med_conf_words):
            return 0.6
        
        # Wenn nur eine Zahl zurückgegeben wird (einfache Antwort)
        if re.match(r'^\d+$', text.strip()):
            return 0.8
        
        # Standard Konfidenz
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
                'error_details': error_msg
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Modellinformationen zurück"""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'framework': 'Transformers/LLaVA',
            'hf_model_name': self.model_name_hf,
            'confidence_threshold': self.confidence_threshold,
            'task': 'person_detection_via_llm',
            'input_size': 'variable',
            'classes': ['person'],
            'device': str(self.model.device) if hasattr(self, 'model') and hasattr(self.model, 'device') else 'unknown'
        }