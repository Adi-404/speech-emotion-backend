from flask import Flask, request, jsonify
from tensorflow.keras.models import model_from_json
import numpy as np
import librosa
import pickle
import os
from dotenv import load_dotenv
import speech_recognition as sr
from google import genai
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
json_file_path = 'CNN_model.json'
weights_file_path = 'best_model1_weights.keras'
scaler_path = 'scaler2.pickle'
encoder_path = 'encoder2.pickle'

# Load model architecture
with open(json_file_path, 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load weights into the model
loaded_model.load_weights(weights_file_path)

# Load scaler and encoder
with open(scaler_path, 'rb') as f:
    scaler2 = pickle.load(f)

with open(encoder_path, 'rb') as f:
    encoder2 = pickle.load(f)

# Emotion labels mapping
emotions1 = {1: 'Neutral', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Fear', 7: 'Disgust', 8: 'Surprise'}

# Feature extraction functions
def zcr(data, frame_length, hop_length):
    return np.squeeze(librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_features = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_features.T) if not flatten else np.ravel(mfcc_features.T)

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    s_rate = 22050
    res=extract_features(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,2376))
    i_result = scaler2.transform(result)
    final_result=np.expand_dims(i_result, axis=2)
    
    return final_result

# Emotion prediction
def predict_emotion(audio_path):
    res = get_predict_feat(audio_path)
    predictions = loaded_model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    return y_pred[0][0]

# Speech-to-text conversion
def convert_wav_to_text(wav_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"

# Gemini API integration
# Load environment variables from .env file
load_dotenv()

# Retrieve API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

def get_gemini_response(user_query, user_emotion):
    mental_health_prompt = f"""
    You are an experienced mental health specialist. Your role is to provide thoughtful, professional, and empathetic responses to users seeking mental health guidance.

    Instructions:
    - Clearly state what emotion the user appears to be feeling.
    - Address the user kindly and with understanding.
    - Acknowledge the user's emotional state.
    - Offer practical advice tailored to the user's emotion, but do not diagnose.
    - Use simple, reassuring language.
    - Adapt your response based on the user's emotional state.

    User's Emotion: {user_emotion}
    User's Query: {user_query}

    Provide a professional, structured response as a mental health expert.
    """

    response = client.models.generate_content(
        model='gemini-2.0-flash-thinking-exp',
        contents=mental_health_prompt,
    )
    return response.text

# API route for processing uploaded audio files
@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Securely save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Process the audio file
    emotion = predict_emotion(file_path)
    transcription = convert_wav_to_text(file_path)
    gemini_response = get_gemini_response(transcription, emotion)

    # Remove processed file
    os.remove(file_path)

    return jsonify({
        "transcription": transcription,
        "emotion": emotion,
        "gemini_response": gemini_response
    })

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)