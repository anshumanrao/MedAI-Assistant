from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# --- MODEL LOADING (Safe Path) ---
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Agar model 'models' folder mein hai toh ye sahi hai, warna path badlein
    model_path = os.path.join(base_path, 'svc.pkl') 
    svc = pickle.load(open(model_path, 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")

# --- FULL MEDICAL DATABASE ---
medical_database = {
    'Fungal infection': {'precautions': ['Keep skin dry', 'Wash towels daily'], 'diet': ['Probiotics', 'Garlic'], 'medication': 'Anti-fungal cream', 'advice': 'Keep area ventilated.'},
    'Allergy': {'precautions': ['Avoid dust', 'Wear mask'], 'diet': ['Ginger tea', 'Honey'], 'medication': 'Cetirizine', 'advice': 'Steam inhalation helps.'},
    'GERD': {'precautions': ['Avoid lying down after meals'], 'diet': ['Cold milk', 'Bananas'], 'medication': 'Antacids', 'advice': 'Eat 3 hours before sleep.'},
    'Malaria': {'precautions': ['Use mosquito nets'], 'diet': ['Coconut water'], 'medication': 'Chloroquine', 'advice': 'Stay hydrated.'},
    'Typhoid': {'precautions': ['Drink boiled water'], 'diet': ['Soft food'], 'medication': 'Antibiotics', 'advice': 'Complete full course.'},
    'Diabetes': {'precautions': ['Regular exercise'], 'diet': ['Low sugar', 'Fiber'], 'medication': 'Insulin/Metformin', 'advice': 'Monitor blood sugar.'}
}

# --- COMPLETE SYMPTOMS LIST (Essential for 132 length vector) ---
# Tip: Aapke dataset mein jo columns hain, unhe lower case mein yahan hona chahiye
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'fatigue': 14,
    'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
    'weight_loss': 19, 'restlessness': 20, 'lethargy': 21
    # Note: Baaki bache huye 132 tak yahan add karna zaroori hai Presentation se pehle!
}

# --- ID TO DISEASE MAP ---
id_map = {
    0: 'Fungal infection', 1: 'Allergy', 2: 'GERD', 3: 'Chronic cholestasis',
    4: 'Drug Reaction', 7: 'Diabetes', 10: 'Hypertension', 14: 'Jaundice',
    15: 'Malaria', 17: 'Dengue', 18: 'Typhoid'
}

@app.route("/")
def index():
    return render_template("index.html", all_symptoms=sorted(symptoms_dict.keys()))

@app.route("/predict", methods=['POST'])
def predict():
    selected = request.form.getlist('symptoms_list')
    if not selected:
        return render_template("index.html", all_symptoms=sorted(symptoms_dict.keys()), error="Please select symptoms!")

    # 1. Input vector of exactly 132 zeros
    input_vector = np.zeros(132) 
    
    # 2. Map selected symptoms to 1s
    for s in selected:
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1

    try:
        # 3. Predict
        prediction_id = svc.predict([input_vector])[0]
        result = id_map.get(prediction_id, "Common Viral/Infection")
    except Exception as e:
        print(f"Prediction Error: {e}")
        result = "Diagnosis Inconclusive"

    # 4. Fetch Details
    data = medical_database.get(result, {
        'precautions': ['Rest well', 'Hydrate'],
        'diet': ['Healthy home food'],
        'medication': 'Consult a GP',
        'advice': 'Further clinical tests required.'
    })

    return render_template("index.html", 
                           all_symptoms=sorted(symptoms_dict.keys()), 
                           result=result, 
                           advice=data.get('advice'),
                           precautions=data.get('precautions'),
                           diet=data.get('diet'),
                           medication=data.get('medication'),
                           selected_symptoms=selected)
