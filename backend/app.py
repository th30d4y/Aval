from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model_data = joblib.load('./models/women_safety_complete_model.joblib')
    classifier = model_data['classifier']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    feature_columns = model_data['feature_columns']
    emergency_keywords = model_data.get('emergency_keywords', {})
    logger.info("✅ Model loaded successfully")
    logger.info(f"📊 Feature columns: {feature_columns}")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    classifier = None

@app.route('/')
def home():
    """Root endpoint with API information"""
    return jsonify({
        'message': 'Women Safety ML API',
        'status': 'running',
        'endpoints': {
            'predict': '/predict',
            'health': '/health'
        },
        'model_loaded': classifier is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if classifier is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        features = data.get('features', {})
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Map area type to encoded value
        area_map = {'safe': 0, 'risky': 1, 'isolated': 2}
        if 'area_type' in features:
            if isinstance(features['area_type'], str):
                features['area_encoded'] = area_map.get(features['area_type'].lower(), 0)
            else:
                features['area_encoded'] = int(features['area_type'])
        
        # Prepare feature array in correct order
        feature_array = []
        for col in feature_columns:
            if col == 'area_encoded':
                feature_array.append(features.get('area_encoded', 0))
            elif col == 'time_hour':
                feature_array.append(features.get('time_hour', datetime.now().hour))
            elif col == 'day_of_week':
                feature_array.append(features.get('day_of_week', datetime.now().weekday()))
            elif col == 'lighting':
                feature_array.append(features.get('lighting', 3))
            elif col == 'crowd_level':
                feature_array.append(features.get('crowd_level', 3))
            elif col == 'emergency_words':
                feature_array.append(features.get('emergency_words', 0))
            elif col == 'voice_distress':
                feature_array.append(features.get('voice_distress', 0.0))
            elif col == 'voice_pitch':
                feature_array.append(features.get('voice_pitch', 150))
            elif col == 'voice_energy':
                feature_array.append(features.get('voice_energy', 0.1))
            elif col == 'fear_level':
                feature_array.append(features.get('fear_level', 0.0))
            elif col == 'help_gesture':
                feature_array.append(features.get('help_gesture', 0.0))
            elif col == 'location_isolation':
                feature_array.append(features.get('location_isolation', 0.3))
            else:
                feature_array.append(features.get(col, 0))
        
        # Scale and predict
        input_scaled = scaler.transform([feature_array])
        prediction = classifier.predict(input_scaled)[0]
        probabilities = classifier.predict_proba(input_scaled)[0]
        
        risk_levels = ['SAFE', 'MEDIUM RISK', 'HIGH RISK']
        
        # ✅ FIX: Convert all numpy types to Python native types
        response = {
            'prediction': int(prediction),  # Convert numpy.int64 to int
            'risk_level': risk_levels[int(prediction)],
            'probabilities': [float(p) for p in probabilities],  # Convert numpy.float64 to float
            'confidence': float(np.max(probabilities)),  # Convert to Python float
            'alert_required': bool(prediction >= 1),  # Ensure Python bool
            'emergency_response': bool(prediction >= 2),  # Ensure Python bool
            'timestamp': datetime.now().isoformat(),
            'features_used': len(feature_array)
        }
        
        # Add recommendations based on prediction
        if prediction == 0:
            response['recommendation'] = "Area appears safe. Stay alert and aware of surroundings."
        elif prediction == 1:
            response['recommendation'] = "Exercise caution. Consider moving to a safer location or seeking company."
        else:
            response['recommendation'] = "HIGH RISK! Seek immediate help and move to safety. Contact emergency services if needed."
        
        logger.info(f"Prediction: {risk_levels[int(prediction)]} (confidence: {float(np.max(probabilities)):.2f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': 'Prediction failed due to an internal error.'}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'feature_count': len(feature_columns) if feature_columns else 0,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/features')
def get_features():
    """Get model feature information"""
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'feature_columns': feature_columns,
        'emergency_keywords': emergency_keywords,
        'model_info': {
            'type': 'GradientBoostingClassifier',
            'classes': ['SAFE', 'MEDIUM RISK', 'HIGH RISK']
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Women's Safety API Server Starting...")
    print("🔗 API Endpoints:")
    print("   • Root: http://localhost:5000/")
    print("   • Predict: http://localhost:5000/predict")
    print("   • Health: http://localhost:5000/health")
    print("   • Features: http://localhost:5000/features")
    print("💡 Ready to receive requests!")
    app.run(host='0.0.0.0', port=5000, debug=True)
