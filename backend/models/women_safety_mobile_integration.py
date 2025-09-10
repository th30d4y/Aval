
# Complete Mobile Integration Code
import joblib
import numpy as np
import librosa

class WomenSafetyMobileApp:
    def __init__(self, model_path):
        """Initialize the mobile app safety analyzer"""
        self.model_data = joblib.load(model_path)
        self.classifier = self.model_data['classifier']
        self.scaler = self.model_data['scaler']
        self.feature_columns = self.model_data['feature_columns']
        self.emergency_keywords = self.model_data['emergency_keywords']
    
    def analyze_safety_with_audio(self, audio_file_path, location_context):
        """Analyze safety using audio input and location context"""
        
        # Extract voice features
        audio_data, sr = librosa.load(audio_file_path, sr=16000)
        
        # Basic voice analysis
        pitches = librosa.yin(audio_data, fmin=80, fmax=400)
        pitch_mean = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 150
        
        rms = librosa.feature.rms(y=audio_data)[0]
        energy_level = np.mean(rms)
        
        # Voice distress calculation
        voice_distress = 0
        if pitch_mean > 220:
            voice_distress += 0.4
        if energy_level > 0.5:
            voice_distress += 0.3
        
        # Prepare features
        area_map = {'safe': 0, 'risky': 1, 'isolated': 2}
        features = [
            area_map.get(location_context.get('area_type', 'safe'), 0),
            location_context.get('time_hour', 12),
            location_context.get('day_of_week', 1),
            location_context.get('lighting', 3),
            location_context.get('crowd_level', 3),
            location_context.get('emergency_words', 0),
            voice_distress,
            pitch_mean,
            energy_level,
            location_context.get('fear_level', 0),
            location_context.get('help_gesture', 0),
            location_context.get('location_isolation', 0.3)
        ]
        
        # Predict
        features_scaled = self.scaler.transform([features])
        prediction = self.classifier.predict(features_scaled)[0]
        probability = self.classifier.predict_proba(features_scaled)[0]
        
        risk_levels = ['SAFE', 'MEDIUM RISK', 'HIGH RISK']
        
        return {
            'risk_level': risk_levels[prediction],
            'confidence': float(max(probability)),
            'alert_required': prediction >= 1,
            'voice_distress': voice_distress,
            'emergency_response': prediction >= 2
        }

# Usage in mobile app:
# safety_app = WomenSafetyMobileApp('women_safety_complete_model.joblib')
# result = safety_app.analyze_safety_with_audio('voice_sample.wav', location_data)
