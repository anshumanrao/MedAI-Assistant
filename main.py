from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# --- MODEL LOADING ---
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'models', 'svc.pkl')
    svc = pickle.load(open(model_path, 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")

# --- UPDATED MEDICAL DATABASE (Common Diseases Focus) ---
medical_database = {
    'Fungal infection': {
        'precautions': ['Keep skin dry', 'Wear cotton clothes', 'Wash towels daily'],
        'diet': ['Probiotics (Curd)', 'Garlic', 'Avoid sugar'],
        'medication': 'Anti-fungal creams (Clotrimazole)',
        'avoid': 'Tight clothes and sharing personal items',
        'advice': 'Keep the affected area ventilated.'
    },
    'Allergy': {
        'precautions': ['Avoid dust/pollen', 'Wear a mask', 'Use air purifier'],
        'diet': ['Ginger tea', 'Vitamin C fruits', 'Honey'],
        'medication': 'Antihistamines (Cetirizine)',
        'avoid': 'Cold drinks and known allergens',
        'advice': 'Steam inhalation helps clear nasal passages.'
    },
    'GERD': {
        'precautions': ['Avoid lying down after meals', 'Small meals', 'Weight management'],
        'diet': ['Cold milk', 'Bananas', 'Oatmeal'],
        'medication': 'Antacids or Proton pump inhibitors',
        'avoid': 'Spicy food, Caffeine, and Alcohol',
        'advice': 'Eat at least 3 hours before sleeping.'
    },
    'Typhoid': {
        'precautions': ['Drink boiled water', 'Hand hygiene', 'Avoid street food'],
        'diet': ['High calorie soft food', 'Plenty of fluids', 'Boiled potatoes'],
        'medication': 'Prescribed Antibiotics course',
        'avoid': 'Raw vegetables and unpeeled fruits',
        'advice': 'Complete the full antibiotic course even if feeling better.'
    },
    'Malaria': {
        'precautions': ['Use mosquito nets', 'Apply repellents', 'Wear full sleeves'],
        'diet': ['Papaya leaf juice', 'Coconut water', 'Pulses/Dal'],
        'medication': 'Anti-malarial drugs (Chloroquine)',
        'avoid': 'Oily food and physical exertion',
        'advice': 'Hydration is key during high fever.'
    }
}

# --- SYMPTOMS DICTIONARY (Based on your Excel Screenshot) ---
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'fatigue': 14,
    'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
    'weight_loss': 19, 'restlessness': 20, 'lethargy': 21
}

# --- ID MAP ---
id_map = {
    0: 'Fungal infection', 1: 'Allergy', 2: 'GERD', 3: 'Chronic cholestasis',
    4: 'Drug Reaction', 7: 'Diabetes', 10: 'Hypertension', 14: 'Jaundice',
    15: 'Malaria', 17: 'Dengue', 18: 'Typhoid'
}

@app.route("/")
def index():
    return render_template("index.html", all_symptoms=symptoms_dict.keys())

@app.route("/predict", methods=['POST'])
def predict():
    selected = request.form.getlist('symptoms_list')
    if not selected:
        return render_template("index.html", all_symptoms=symptoms_dict.keys(), error="Select symptoms first!")

    input_vector = np.zeros(132)
    for s in selected:
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1

    try:
        prediction_id = svc.predict([input_vector])[0]
        result = id_map.get(prediction_id, "Common Viral/Cold")
    except:
        result = "Condition Not Found"

    # Get data from database or show default
    data = medical_database.get(result, {
        'precautions': ['Rest well', 'Hydrate'],
        'diet': ['Healthy home food'],
        'medication': 'Consult a Pharmacist/GP',
        'avoid': 'Cold environment and stress',
        'advice': 'Please consult a doctor for a physical exam.'
    })

    return render_template("index.html", 
                           all_symptoms=symptoms_dict.keys(), 
                           result=result, 
                           advice=data.get('advice'),
                           precautions=data.get('precautions'),
                           diet=data.get('diet'),
                           medication=data.get('medication'),
                           avoid=data.get('avoid'),
                           selected_symptoms=selected)

if __name__ == "__main__":
    app.run(debug=True)