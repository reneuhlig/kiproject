from deepface import DeepFace
import cv2
import numpy as np
from typing import Dict, Any, List
from BaseDetector import BaseDetector


class DeepFacePersonDetector(BaseDetector):
    """DeepFace Detektor für Personenerkennung über Gesichtserkennung"""
    
    def __init__(self, detector_backend: str = "opencv", confidence_threshold: float = 0.5):
        """
        Initialisiert den DeepFace Detektor
        
        Args:
            detector_backend: Backend für Gesichtserkennung ('opencv', 'ssd', 'mtcnn', 'retinaface')
            confidence_threshold: Mindest-Konfidenz für Detections
        """
        super().__init__("DeepFace", "2.0")
        self.detector_backend = detector_backend
        self.confidence_threshold = confidence_threshold
        
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Erkennt Personen über Gesichtserkennung
        
        Args:
            image_path: Pfad zum Bild
            
        Returns:
            Dictionary mit Erkennungsergebnissen
        """
        try:
            # Erst versuchen Gesichter zu extrahieren (einfacher Test)
            faces = []
            try:
                faces = DeepFace.extract_faces(
                    img_path=image_path,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    align=True
                )
            except Exception as e:
                print(f"Warnung: Gesichtsextraktion fehlgeschlagen: {e}")
                # Fallback zu anderem Backend
                if self.detector_backend != 'opencv':
                    try:
                        faces = DeepFace.extract_faces(
                            img_path=image_path,
                            detector_backend='opencv',
                            enforce_detection=False,
                            align=True
                        )
                    except Exception as e2:
                        print(f"Auch OpenCV Backend fehlgeschlagen: {e2}")
            
            # Für jedes Gesicht auch die Region ermitteln
            face_objs = []
            confidences = []
            
            if faces and len(faces) > 0:
                try:
                    # Detaillierte Analyse für Bounding Boxes
                    analysis = DeepFace.analyze(
                        img_path=image_path,
                        detector_backend=self.detector_backend,
                        enforce_detection=False,
                        actions=['age'],  # Minimale Analyse für Performance
                        silent=True
                    )
                    
                    # Behandle sowohl einzelne Ergebnisse als auch Listen
                    if not isinstance(analysis, list):
                        analysis = [analysis]
                    
                    for face_info in analysis:
                        region = face_info.get('region', {})
                        if region and all(k in region for k in ['x', 'y', 'w', 'h']):
                            # Konfidenz basierend auf Gesichtsqualität schätzen
                            confidence = self._estimate_face_confidence(faces, region)
                            
                            if confidence >= self.confidence_threshold:
                                face_objs.append({
                                    'bbox': [
                                        region['x'], 
                                        region['y'], 
                                        region['x'] + region['w'], 
                                        region['y'] + region['h']
                                    ],
                                    'confidence': confidence,
                                    'class': 'person',
                                    'face_region': region
                                })
                                confidences.append(confidence)
                                
                except Exception as e:
                    # Fallback: Nur Anzahl der extrahierten Gesichter
                    print(f"Detaillierte Analyse fehlgeschlagen: {e}")
                    for i, face in enumerate(faces):
                        if face is not None and face.shape[0] > 0 and face.shape[1] > 0:
                            confidence = self._estimate_face_confidence_simple(face)
                            if confidence >= self.confidence_threshold:
                                face_objs.append({
                                    'bbox': [0, 0, 100, 100],  # Placeholder
                                    'confidence': confidence,
                                    'class': 'person',
                                    'face_index': i
                                })
                                confidences.append(confidence)
            
            # Ergebnis zusammenstellen
            return {
                'persons_detected': len(face_objs),
                'persons': face_objs,
                'confidences': confidences,
                'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
                'max_confidence': float(max(confidences)) if confidences else 0.0,
                'min_confidence': float(min(confidences)) if confidences else 0.0,
                'uncertain': any(c < 0.7 for c in confidences) if confidences else len(faces) > 0,
                'model_output': {
                    'total_faces_extracted': len(faces),
                    'valid_detections': len(face_objs),
                    'detector_backend': self.detector_backend
                }
            }
            
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
                'model_output': {'error_details': str(e)}
            }
    
    def _estimate_face_confidence(self, faces: List, region: Dict) -> float:
        """Schätzt Konfidenz basierend auf Gesichtsregion"""
        try:
            # Basiskonfidenz
            confidence = 0.8
            
            # Reduziere basierend auf Gesichtsgröße
            face_area = region.get('w', 0) * region.get('h', 0)
            if face_area < 1000:  # Sehr kleine Gesichter
                confidence *= 0.6
            elif face_area < 5000:  # Kleine Gesichter
                confidence *= 0.8
                
            return min(confidence, 0.95)
        except:
            return 0.7
    
    def _estimate_face_confidence_simple(self, face_array: np.ndarray) -> float:
        """Einfache Konfidenzschätzung basierend auf Gesichtsdaten"""
        try:
            # Basiere auf Bildqualität
            if face_array.shape[0] < 50 or face_array.shape[1] < 50:
                return 0.5
            
            # Prüfe Bildschärfe (einfache Varianz-basierte Methode)
            if len(face_array.shape) == 3:
                gray = cv2.cvtColor((face_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = (face_array * 255).astype(np.uint8)
                
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if variance > 500:
                return 0.8
            elif variance > 100:
                return 0.6
            else:
                return 0.4
                
        except:
            return 0.6
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Modellinformationen zurück"""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'framework': 'DeepFace',
            'detector_backend': self.detector_backend,
            'confidence_threshold': self.confidence_threshold,
            'task': 'person_detection_via_faces',
            'input_size': 'variable',
            'classes': ['person']
        }