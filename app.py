import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
from flask import Flask, render_template, request, make_response, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D  # type: ignore
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from xhtml2pdf import pisa  # type: ignore
from io import BytesIO

# === Flask Configuration ===
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session to work
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === Rebuild Model and Load Weights ===
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='softmax')
])

model.load_weights("skin_disease_150_model.keras")

# === Load Disease Information ===
with open("data/disease_info.json") as f:
    disease_data = json.load(f)

# === Class Labels (must match model) ===
class_labels = [
    "Cellulitis", "Impetigo", "Athlete's Foot",
    "Nail Fungus", "Ringworm", "Cutaneous Larva Migrans",
    "Chickenpox", "Shingles"
]

# === Routes ===

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questions')
def questions():
    return render_template('questions.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    name = request.form['name']
    age = request.form['age']
    address = request.form['address']
    mobile = request.form['mobile']
    symptoms = request.form['symptoms']
    dt_str = request.form['datetime']
    date_time = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M")

    image = request.files['image']
    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    img = Image.open(filepath).resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)

    predicted_disease = class_labels[predicted_index]
    prediction_score = float(round(predictions[predicted_index] * 100, 2))

    detail = disease_data[predicted_disease]["detail"]
    precautions = disease_data[predicted_disease]["precautions"]
    treatment = disease_data[predicted_disease]["treatment"]

    # Store data in session for PDF generation
    session['result_data'] = {
        "name": name,
        "age": age,
        "address": address,
        "mobile": mobile,
        "symptoms": symptoms,
        "date": date_time.strftime("%Y-%m-%d"),
        "time": date_time.strftime("%H:%M"),
        "image_filename": filename,
        "predicted_disease": predicted_disease,
        "prediction_score": prediction_score,
        "disease_detail": detail,
        "disease_precautions": precautions,
        "disease_treatment": treatment
    }

    return render_template('result.html',
                           name=name,
                           age=age,
                           address=address,
                           mobile=mobile,
                           date=date_time.strftime("%Y-%m-%d"),
                           time=date_time.strftime("%H:%M"),
                           image_filename=filename,
                           predicted_disease=predicted_disease,
                           prediction_score=prediction_score,
                           disease_detail=detail,
                           disease_precautions=precautions,
                           disease_treatment=treatment)

@app.route('/generate-pdf')
def generate_pdf():
    if 'result_data' not in session:
        return "No result data available. Please analyze an image first.", 400

    data = session['result_data']
    data['image_path'] = os.path.join("static", "uploads", data['image_filename'])

    rendered_html = render_template('report_template.html', **data)

    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(rendered_html, dest=pdf)

    if pisa_status.err:
        return "Failed to generate PDF", 500

    response = make_response(pdf.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=diagnosis_report.pdf'
    return response

# === Run the Flask App ===
if __name__ == '__main__':
    app.run(debug=True)
