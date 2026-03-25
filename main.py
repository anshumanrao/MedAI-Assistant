from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# --- 1. MODEL LOADING ---
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'svc.pkl') 
    if not os.path.exists(model_path):
        model_path = os.path.join(base_path, 'models', 'svc.pkl')
        
    svc = pickle.load(open(model_path, 'rb'))
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- 2. SYMPTOMS DICTIONARY (132 Symptoms) ---
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
    'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
    'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_urination': 13, 'fatigue': 14,
    'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19,
    'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29,
    'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34,
    'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39,
    'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49,
    'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
    'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64,
    'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
    'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79,
    'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84,
    'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
    'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94,
    'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99,
    'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic_patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104,
    'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109,
    'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114,
    'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
    'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
    'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129,
    'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

# --- 3. MEDICAL DATABASE ---
medical_database = {
    'Fungal infection': {'precautions': ['Keep skin dry', 'Wash towels daily'], 'diet': ['Garlic', 'Probiotics'], 'medication': 'Anti-fungal cream', 'advice': 'Keep area ventilated.'},
    'Allergy': {'precautions': ['Avoid dust', 'Wear mask'], 'diet': ['Ginger tea', 'Honey'], 'medication': 'Cetirizine', 'advice': 'Steam inhalation helps.'},
    'GERD': {'precautions': ['Avoid lying down after meals'], 'diet': ['Cold milk', 'Bananas'], 'medication': 'Antacids', 'advice': 'Eat 3 hours before sleep.'},
    'Diabetes': {'precautions': ['Regular exercise', 'Monitor sugar'], 'diet': ['Low sugar', 'Fiber'], 'medication': 'Insulin/Metformin', 'advice': 'Stay active.'},
    'Hypertension': {'precautions': ['Reduce salt', 'Manage stress'], 'diet': ['Fruits', 'Vegetables'], 'medication': 'BP meds', 'advice': 'Regular BP check.'}
}

@app.route("/")
def index():
    return render_template("index.html", all_symptoms=sorted(symptoms_dict.keys()))

@app.route("/predict", methods=['POST'])
def predict():
    selected = request.form.getlist('symptoms_list')
    if not selected:
        return render_template("index.html", all_symptoms=sorted(symptoms_dict.keys()), error="Select symptoms first!")

    # A. Input Vector
    input_vector = np.zeros(132)
    for s in selected:
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1

    # B. Prediction Logic (Optimized for String Output)
    try:
        prediction = svc.predict([input_vector])[0]
        print(f"🛠️ DEBUG: Model predicted: {prediction}")
        
        # Agar model seedha naam de raha hai
        result = str(prediction) 
    except Exception as e:
        print(f"❌ DEBUG Error: {e}")
        result = "Condition Not Found"

    # C. Details Fetching
    data = medical_database.get(result, {
        'precautions': ['Rest well', 'Stay hydrated'],
        'diet': ['Healthy home food'],
        'medication': 'Consult a GP',
        'advice': 'Please visit a doctor for a physical exam.'
    })

    return render_template("index.html", 
                           all_symptoms=sorted(symptoms_dict.keys()), 
                           result=result, 
                           advice=data.get('advice'),
                           precautions=data.get('precautions'),
                           diet=data.get('diet'),
                           medication=data.get('medication'),
                           selected_symptoms=selected)

if __name__ == "__main__":
    app.run(debug=True)
