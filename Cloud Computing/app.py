from flask import Flask, request, jsonify
import tensorflow as tf
print(tf.__version__)
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging
from tensorflow import keras
from io import BytesIO
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcs_key.json"  

app = Flask(__name__)

# Konfigurasi GCP
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GCS_KEY_JSON", "gcs_key.json")
#BUCKET_NAME = os.getenv("BUCKET_NAME", "nutrise-model")
#MODEL_FILE = os.getenv("MODEL_FILE", "model.h5")

BUCKET_NAME = 'nutrise-model' 
MODEL_FILE = 'model.h5'

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Konfigurasi Firebase
cred = credentials.Certificate("firebase_key.json") 
firebase_admin.initialize_app(cred, {'projectId': 'nutrise-444515'})
db = firestore.client(database_id="model-data")

# Fungsi untuk load model dari Cloud Storage
def load_model_from_gcs():
    try:
        logging.info("Loading model from GCS...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILE)
        #blob.download_to_filename(MODEL_FILE)

        local_model_path = '/tmp/model.h5'
        blob.download_to_filename(local_model_path)

        logging.info("Loading model...")
        model = tf.keras.models.load_model(local_model_path, custom_objects={'mse': mse}) 
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

# Load model
model = load_model_from_gcs()

# API endpoint untuk prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Ubah FileStorage menjadi BytesIO
    img_bytes = BytesIO(file.read())

    # Gunakan BytesIO sebagai input ke load_img
    img = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(224, 224))

    #  Preprocessing gambar (sesuaikan dengan model Anda)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    img_array /= 255.0  # Normalisasi

    # Prediksi
    predictions = model.predict(img_array)
    if isinstance(predictions, list):
        predictions = predictions[0]
    predicted_class = predictions.argmax()

    #  Ambil data dari Firebase berdasarkan predicted_class (sesuaikan dengan struktur data Anda)
    doc_ref = db.collection('makanan').document(str(predicted_class))
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        return jsonify(data)
    else:
        return jsonify({'error': 'Data not found in Firebase'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)