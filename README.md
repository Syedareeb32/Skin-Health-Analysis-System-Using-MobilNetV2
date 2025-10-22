# Skin-Health-Analysis-System-Using-MobilNetV2
A Flask-based AI web app for automated dermatological diagnosis. 🔹 Processes skin images, predicts 8 disease types, and provides treatment insights.

🧠 Intelligent Skin Disease Detection using CNNs

An AI-driven Flask web application for automated skin disease detection and classification using MobileNetV2 and TensorFlow.
This system analyzes dermatological images, predicts the disease class, and provides medical details, precautions, and treatment insights.

🚀 Overview

This project integrates Deep Learning (MobileNetV2) with a Flask-based web interface to perform real-time skin disease analysis.
The model classifies input images into eight categories — Acne, Eczema, Melanoma, Psoriasis, Ringworm, Rosacea, Shingles, and Vitiligo.
After analysis, the system displays the predicted disease name, confidence score, and generates a downloadable PDF report for patient documentation.

This project bridges the gap between AI and healthcare, enabling accessible dermatological screening through an interactive web platform.

⚙️ Features

🧠 Deep Learning model (MobileNetV2) trained on 10,000+ images

🌐 Flask-based interactive web interface

🧩 Real-time disease prediction and confidence scoring

📄 Automatic PDF report generation

💊 Detailed disease information, causes, and treatments

🖼️ Image upload and preprocessing (224×224 RGB normalization)

🧰 Tech Stack
Category	Tools & Libraries
Frontend	HTML, CSS, JavaScript
Backend	Flask (Python)
Deep Learning	TensorFlow, Keras, NumPy
Utilities	Pillow (PIL), xhtml2pdf, JSON
Model	MobileNetV2 (Pretrained on ImageNet)
🧪 How to Run Locally
# 1️⃣ Clone the repository
git clone https://github.com/your-username/skin-disease-detection.git

# 2️⃣ Navigate to the project directory
cd skin-disease-detection

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the Flask app
python app.py


Then open your browser and go to 👉 http://127.0.0.1:5000/

🧬 Model Details

Architecture: MobileNetV2 (transfer learning)

Input Size: 224×224×3 RGB images

Output: 8 disease classes

Optimizer: Adam

Loss Function: Categorical Crossentropy

Accuracy: Achieved high accuracy on test dataset

📘 Project Structure
skin-disease-detection/
│
├── app.py                     # Flask application
├── skin_disease_150_model.keras  # Trained CNN model
├── data/
│   └── disease_info.json       # Disease descriptions, causes, treatments
├── static/
│   └── uploads/                # Uploaded images
├── templates/
│   ├── index.html
│   ├── upload.html
│   ├── result.html
│   └── report_template.html
├── requirements.txt
└── README.md

🧾 Author

Syed Areeb Ali Shah
📧 21it31@quest.edu.pk |alt: sareeb608@gmail.com
]
💼 www.linkedin.com/in/syed-areeb-ali-shah-646a79241

💡 Future Enhancements

Integrate symptom-based diagnosis form

Add more disease classes

Deploy on cloud (AWS / Render / Hugging Face Spaces)

Include multi-language report generation

🎓 Acknowledgment

Special thanks to MobileNetV2 architecture developers and the TensorFlow community for enabling accessible and efficient AI model deployment.
