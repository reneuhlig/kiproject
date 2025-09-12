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
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
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
            # Bild laden
            image = Image.open(image_path).convert('RGB')
            
            # Prompt für Personenzählung
            prompt = """USER: <image>
Look at this image carefully. Count the number of people you can see in the image. 
Provide your answer in the following JSON format:
{
    "person_count": <number>,
    "confidence": <confidence_score_0_to_1>,
    "description": "<brief description of what you see>",
    "uncertain_areas": "<any areas where you're unsure>"
}

Be precise with counting. If you're unsure about some people (partially visible, unclear, etc.), mention that in uncertain_areas and adjust your confidence accordingly.

ASSISTANT: """
            
            # Input vorbereiten
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.model.device)
            
            # Generation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Response dekodieren
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extrahiere nur den Assistant-Teil
            assistant_response = response.split("ASSISTANT: ")[-1].strip()
            
            # Parse die Antwort
            result = self._parse_response(assistant_response)
            
            return result
            
        except Exception as e:
            return {
                'persons_detected': 0,
                'persons': [],
                'confidences': [],
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'uncertain': True,
                'error': str(e),
                'model_output': {'raw_error': str(e)}
            }
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parst die Modellantwort"""
        try:
            # Versuche JSON zu extrahieren
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                person_count = parsed.get('person_count', 0)
                confidence = parsed.get('confidence', 0.5)
                description = parsed.get('description', '')
                uncertain_areas = parsed.get('uncertain_areas', '')
                
            else:
                # Fallback: Suche nach Zahlen im Text
                numbers = re.findall(r'\b(\d+)\b', response)
                person_count = int(numbers[0]) if numbers else 0
                
                # Schätze Konfidenz basierend auf Schlüsselwörtern
                confidence = self._estimate_confidence_from_text(response)
                description = response[:100] + "..." if len(response) > 100 else response
                uncertain_areas = "Parsed from unstructured response"
            
            # Erstelle einheitliche Person-Objekte
            persons = []
            confidences = []
            
            if person_count > 0:
                # Da wir keine genauen Bounding Boxes haben, erstelle generische Einträge
                for i in range(person_count):
                    person_confidence = max(0.1, confidence + ((-0.1) * i * 0.1))  # Leicht reduzierte Konfidenz für weitere Personen
                    persons.append({
                        'bbox': [0, 0, 100, 100],  # Placeholder bbox
                        'confidence': person_confidence,
                        'class': 'person',
                        'detection_method': 'llm_analysis'
                    })
                    confidences.append(person_confidence)
            
            return {
                'persons_detected': person_count,
                'persons': persons,
                'confidences': confidences,
                'avg_confidence': confidence,
                'max_confidence': max(confidences) if confidences else 0.0,
                'min_confidence': min(confidences) if confidences else 0.0,
                'uncertain': confidence < 0.7 or len(uncertain_areas.strip()) > 0,
                'model_output': {
                    'raw_response': response,
                    'description': description,
                    'uncertain_areas': uncertain_areas,
                    'llm_confidence': confidence
                }
            }
            
        except Exception as e:
            # Kompletter Fallback
            return {
                'persons_detected': 0,
                'persons': [],
                'confidences': [],
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'uncertain': True,
                'error': f"Parse error: {str(e)}",
                'model_output': {'raw_response': response, 'parse_error': str(e)}
            }
    
    def _estimate_confidence_from_text(self, text: str) -> float:
        """Schätzt Konfidenz basierend auf Textinhalt"""
        text_lower = text.lower()
        
        # Hohe Konfidenz Indikatoren
        if any(word in text_lower for word in ['clearly', 'obviously', 'definitely', 'certainly']):
            return 0.9
        
        # Niedrige Konfidenz Indikatoren
        if any(word in text_lower for word in ['maybe', 'possibly', 'might', 'unclear', 'difficult', 'hard to']):
            return 0.4
        
        # Mittlere Konfidenz Indikatoren
        if any(word in text_lower for word in ['appears', 'seems', 'likely', 'probably']):
            return 0.6
        
        # Standard Konfidenz
        return 0.7
    
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
            'device': str(self.model.device) if hasattr(self, 'model') else 'unknown'
        }