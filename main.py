from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import difflib

# Load data
symptom_severity = pd.read_csv("Symptom-severity.csv")
symptom_severity['Symptom'] = symptom_severity['Symptom'].str.lower()
severity_dict = dict(zip(symptom_severity["Symptom"], symptom_severity["weight"]))

# Flask app
app = Flask(__name__)

# Load other datasets
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv("medications.csv")
diets = pd.read_csv("diets.csv")

# Load trained model
svc = pickle.load(open('svc_final.pkl', 'rb'))

# Dictionary mapping symptoms to indices
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Synonym mapping for common user-friendly terms
synonym_map = {
    "no_appetite": "loss_of_appetite",
    "lack_of_appetite": "loss_of_appetite",
    "appetite_loss": "loss_of_appetite",
    "feel_like_vomiting": "vomiting",
    "throwing_up": "vomiting",
    "sick_to_stomach": "vomiting",
    "tired": "fatigue",
    "weak": "fatigue",
    "lethargic": "lethargy",
    "loose_motions": "diarrhoea",
    "stomach_upset": "indigestion",
    "bellyache": "abdominal_pain",
    "pain_in_belly": "abdominal_pain",
    "head_pain": "headache",
    "dizzy": "dizziness",
    "no_energy": "fatigue",
    "dry_cough": "cough",
    "wet_cough": "cough",
    "cold": "runny_nose",
    "sore_throat": "throat_irritation",
    "pain_while_urinating": "burning_micturition",
    "urine_burning": "burning_micturition",
    "dark_colored_urine": "dark_urine",
    "pain_while_pooping": "pain_during_bowel_movements",
    "constipated": "constipation",
    "upset_stomach": "stomach_pain",
    "eye_pain": "pain_behind_the_eyes",
    "nauseated": "nausea",
    "tummy_pain": "belly_pain",
    "face_swollen": "puffy_face_and_eyes",
    "urine_smell": "foul_smell_of urine",
    "bloating": "distention_of_abdomen",
    "bloated": "distention_of_abdomen",
    "gas": "passage_of_gases",
    "gas_pain": "passage_of_gases",
    "red_eyes": "redness_of_eyes",
    "eye_redness": "redness_of_eyes",
    "frequent_urination": "polyuria",
    "swollen_legs": "swollen_legs",
    "itchy_skin": "itching",
    "skin_peel": "skin_peeling",
    "yellow_skin": "yellowish_skin",
    "yellow_eyes": "yellowing_of_eyes",
    "joint_ache": "joint_pain",
    "body_ache": "muscle_pain"
}

def standardize_symptoms(user_symptoms):
    standardized = []
    for symptom in user_symptoms:
        key = symptom.strip().lower().replace(" ", "_")
        mapped = synonym_map.get(key, key)
        standardized.append(mapped)
    return standardized


# Helper to find closest symptoms
def find_similar_symptoms(invalid_symptom):
    valid_symptoms = list(symptoms_dict.keys())
    return difflib.get_close_matches(invalid_symptom, valid_symptoms, n=5, cutoff=0.6)

# Helper to get info
def helper(dis):
    desc = " ".join(description[description['Disease'] == dis]['Description'].values)
    pre = precautions[precautions['Disease'] == dis].iloc[:, 1:5].values.flatten()
    pre = [p for p in pre if pd.notna(p)]

    med = medications[medications['Disease'] == dis]['Medication'].values
    med = [m.strip() for m in med[0].split(',')] if len(med) > 0 else []

    die = diets[diets['Disease'] == dis]['Diet'].values
    die = [d.strip() for d in die[0].split(',')] if len(die) > 0 else []

    wrkout = workout[workout['disease'] == dis]['workout'].values
    wrkout = [w.strip() for w in wrkout[0].split(',')] if len(wrkout) > 0 else []

    return desc, pre, med, die, wrkout

# Predict function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        symptom = symptom.strip().lower().replace(" ", "_")
        if symptom in symptoms_dict:
            idx = symptoms_dict[symptom]
            weight = severity_dict.get(symptom, 1)
            input_vector[idx] = weight
    prediction = svc.predict([input_vector])[0]
    return diseases_list.get(prediction, None)

# Home route
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if not symptoms or symptoms.strip().lower() == "symptoms":
            return render_template('index.html', message="Please enter valid symptoms separated by commas.")

        #user_symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms.split(',')]
        raw_symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms.split(',')]
        user_symptoms = standardize_symptoms(raw_symptoms)

        valid_input = [s for s in user_symptoms if s in symptoms_dict]

        predicted_disease = get_predicted_value(valid_input)

        if not predicted_disease:
            # Now check for possible corrections only if prediction failed
            invalid_symptoms = [s for s in user_symptoms if s not in symptoms_dict]
            similar_symptoms = []
            for s in invalid_symptoms:
                similar_symptoms.extend(find_similar_symptoms(s))
            if similar_symptoms:
                message = f"Could not recognize: {', '.join(invalid_symptoms)}. Did you mean: {', '.join(set(similar_symptoms))}?"
            else:
                message = f"Could not recognize symptoms or make a prediction. Please check your input."
            return render_template('index.html', message=message)

        # Fetch associated data
        dis_des, my_precautions, my_medications, my_diet, my_workout = helper(predicted_disease)
        

        return render_template('index.html',
                               predicted_disease=predicted_disease,
                               dis_des=dis_des,
                               my_precautions=my_precautions,
                               medications=my_medications,
                               my_diet=my_diet,
                               workout=my_workout)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
