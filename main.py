# from flask import Flask, request, render_template, jsonify  # Import jsonify
# import numpy as np
# import pandas as pd
# import pickle
# symptom_severity = pd.read_csv("Symptom-severity.csv")
# # Convert to lowercase for matching consistency
# symptom_severity['Symptom'] = symptom_severity['Symptom'].str.lower()
# severity_dict = dict(zip(symptom_severity["Symptom"], symptom_severity["weight"]))



# # flask app
# app = Flask(__name__)

# # load databasedataset===================================
# sym_des = pd.read_csv("symtoms_df.csv")
# precautions = pd.read_csv("precautions_df.csv")
# workout = pd.read_csv("workout_df.csv")
# description = pd.read_csv("description.csv")
# medications = pd.read_csv('medications.csv')
# diets = pd.read_csv("diets.csv")

# # load model===========================================
# svc = pickle.load(open('svc.pkl','rb'))

# #============================================================
# # custome and helping functions
# #==========================helper funtions================
# def helper(dis):
#     desc = description[description['Disease'] == dis]['Description']
#     desc = " ".join([w for w in desc])

#     pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
#     pre = [col for col in pre.values]

#     med = medications[medications['Disease'] == dis]['Medication']
#     med = [med for med in med.values]

#     die = diets[diets['Disease'] == dis]['Diet']
#     die = [die for die in die.values]

#     wrkout = workout[workout['disease'] == dis] ['workout']


#     return desc,pre,med,die,wrkout

# symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
# diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# # Model Prediction function
# '''def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))
#     for item in patient_symptoms:
#         input_vector[symptoms_dict[item]] = 1
#     return diseases_list[svc.predict([input_vector])[0]]'''

# '''def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))
#     for symptom in patient_symptoms:
#         if symptom in symptoms_dict:
#             idx = symptoms_dict[symptom]
#             # Use severity weight instead of just 1
#             weight = symptom_severity[symptom_severity["Symptom"] == symptom]["weight"].values
#             input_vector[idx] = weight[0] if len(weight) > 0 else 1
#     prediction = svc.predict([input_vector])[0]
#     return diseases_list.get(prediction, "Unknown Disease")'''
# def find_similar_symptoms(invalid_symptom):
#     """Find similar symptoms using various matching techniques"""
#     similar_symptoms = []
    
#     # Common word variations mapping with exact matches
#     variations = {
#         'cough': ['cough'],
#         'fever': ['high_fever', 'mild_fever'],
#         'headache': ['headache'],
#         'pain': ['joint_pain', 'stomach_pain', 'back_pain', 'chest_pain', 'neck_pain', 
#                 'knee_pain', 'hip_joint_pain', 'pain_behind_the_eyes', 'pain_in_anal_region', 
#                 'pain_during_bowel_movements'],
#         'stomach': ['stomach_pain', 'stomach_bleeding', 'swelling_of_stomach'],
#         'nausea': ['nausea', 'vomiting'],
#         'fatigue': ['fatigue', 'lethargy', 'malaise', 'weakness_in_limbs', 'weakness_of_one_body_side'],
#         'rash': ['skin_rash', 'nodal_skin_eruptions', 'silver_like_dusting', 'blister', 'pus_filled_pimples', 'red_spots_over_body'],
#         'swelling': ['swelling_joints', 'swollen_legs', 'swelled_lymph_nodes', 'swollen_extremeties', 'swollen_blood_vessels'],
#         'dizziness': ['dizziness', 'spinning_movements', 'loss_of_balance', 'unsteadiness'],
#         'diarrhea': ['diarrhoea', 'bloody_stool'],
#         'constipation': ['constipation'],
#         'anxiety': ['anxiety', 'irritability', 'restlessness'],
#         'depression': ['depression', 'mood_swings'],
#         'insomnia': ['insomnia', 'altered_sensorium'],
#         'backache': ['back_pain'],
#         'jointache': ['joint_pain'],
#         'muscleache': ['muscle_pain', 'muscle_weakness', 'muscle_wasting'],
#         'chestache': ['chest_pain'],
#         'throatache': ['throat_irritation', 'patches_in_throat'],
#         'neckache': ['neck_pain', 'stiff_neck'],
#         'kneeache': ['knee_pain'],
#         'hipache': ['hip_joint_pain'],
#         'legache': ['knee_pain', 'swollen_legs'],
#         'eyeissue': ['redness_of_eyes', 'watering_from_eyes', 'blurred_and_distorted_vision', 'visual_disturbances'],
#         'skinissue': ['skin_rash', 'blister', 'itching', 'skin_peeling', 'dischromic _patches'],
#         'urinaryissue': ['burning_micturition', 'foul_smell_of urine', 'foul_smell_ofurine', 'yellow_urine', 'polyuria'],
#         'vomiting': ['vomiting']
#     }
    
#     # First check for exact matches in variations
#     for key, value in variations.items():
#         if key == invalid_symptom.lower():
#             if isinstance(value, list):
#                 similar_symptoms.extend(value)
#             else:
#                 similar_symptoms.append(value)
#             return similar_symptoms  # Return immediately if exact match found
    
#     # If no exact match, check for partial matches
#     for valid_sym in symptoms_dict.keys():
#         # Remove underscores and convert to lowercase for comparison
#         clean_invalid = invalid_symptom.lower().replace('_', ' ')
#         clean_valid = valid_sym.lower().replace('_', ' ')
        
#         # Check if words are similar
#         if (clean_invalid in clean_valid or 
#             clean_valid in clean_invalid or 
#             any(word in clean_valid for word in clean_invalid.split()) or
#             any(word in clean_invalid for word in clean_valid.split())):
#             similar_symptoms.append(valid_sym)
    
#     return list(set(similar_symptoms))


# def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))

#     for symptom in patient_symptoms:
#         if symptom in symptoms_dict:
#             idx = symptoms_dict[symptom]
#             weight = severity_dict.get(symptom, 1)
#             input_vector[idx] = weight

#     prediction = svc.predict([input_vector])[0]
#     return diseases_list.get(prediction, "Unknown Disease")



# # creating routes========================================


# @app.route("/")
# def index():
#     return render_template("index.html")

# # Define a route for the home page
# @app.route('/predict', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         symptoms = request.form.get('symptoms')
#         print(symptoms)
#         if not symptoms or symptoms.strip().lower() == "symptoms":
#             message = "Please enter valid symptoms separated by commas."
#             return render_template('index.html', message=message)

#         # Split and clean input symptoms
#         user_symptoms = [s.strip().lower() for s in symptoms.split(',')]
#         user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

#         # Check if all symptoms are valid
#         invalid_symptoms = [sym for sym in user_symptoms if sym not in symptoms_dict]
#         if invalid_symptoms:
#             message = f"The following symptoms are not recognized: {', '.join(invalid_symptoms)}"
#             return render_template('index.html', message=message)

#         try:
#             predicted_disease = get_predicted_value(user_symptoms)
#             dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

#             my_precautions = []
#             for i in precautions[0]:
#                 my_precautions.append(i)

#             return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
#                                    my_precautions=my_precautions, medications=medications,
#                                    my_diet=rec_diet, workout=workout)

#         except Exception as e:
#             message = "Sorry, something went wrong during prediction. Please try again."
#             print(f"Error: {e}")
#             return render_template('index.html', message=message)

#     return render_template('index.html')


# if __name__ == '__main__':

#     app.run(debug=True)

from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
import difflib

symptom_severity = pd.read_csv("Symptom-severity.csv")
# # Convert to lowercase for matching consistency
symptom_severity['Symptom'] = symptom_severity['Symptom'].str.lower()
severity_dict = dict(zip(symptom_severity["Symptom"], symptom_severity["weight"]))



# flask app
app = Flask(__name__)

# load databasedataset===================================
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")

# load model===========================================
svc = pickle.load(open('svc.pkl','rb'))

#============================================================
# custome and helping functions
#==========================helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = [w for w in wrkout.values]

    return desc, pre, med, die, wrkout

# Correct symptoms dictionary from the training data
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def find_similar_symptoms(invalid_symptom):
    """Find similar symptoms using various matching techniques"""
    valid_symptoms = list(symptoms_dict.keys())
    matches = difflib.get_close_matches(invalid_symptom, valid_symptoms, n=5, cutoff=0.6)
    
    # Common word variations mapping with exact matches
    variations = {
        'cough': ['cough'],
        'fever': ['high_fever', 'mild_fever'],
        'headache': ['headache'],
        'pain': ['joint_pain', 'stomach_pain', 'back_pain', 'chest_pain', 'neck_pain', 
                'knee_pain', 'hip_joint_pain', 'pain_behind_the_eyes', 'pain_in_anal_region', 
                'pain_during_bowel_movements'],
        'stomach': ['stomach_pain', 'stomach_bleeding', 'swelling_of_stomach'],
        'nausea': ['nausea', 'vomiting'],
        'fatigue': ['fatigue', 'lethargy', 'malaise', 'weakness_in_limbs', 'weakness_of_one_body_side'],
        'rash': ['skin_rash', 'nodal_skin_eruptions', 'silver_like_dusting', 'blister', 'pus_filled_pimples', 'red_spots_over_body'],
        'swelling': ['swelling_joints', 'swollen_legs', 'swelled_lymph_nodes', 'swollen_extremeties', 'swollen_blood_vessels'],
        'dizziness': ['dizziness', 'spinning_movements', 'loss_of_balance', 'unsteadiness'],
        'diarrhea': ['diarrhoea', 'bloody_stool'],
        'constipation': ['constipation'],
        'anxiety': ['anxiety', 'irritability', 'restlessness'],
        'depression': ['depression', 'mood_swings'],
        'insomnia': ['insomnia', 'altered_sensorium'],
        'backache': ['back_pain'],
        'jointache': ['joint_pain'],
        'muscleache': ['muscle_pain', 'muscle_weakness', 'muscle_wasting'],
        'chestache': ['chest_pain'],
        'throatache': ['throat_irritation', 'patches_in_throat'],
        'neckache': ['neck_pain', 'stiff_neck'],
        'kneeache': ['knee_pain'],
        'hipache': ['hip_joint_pain'],
        'legache': ['knee_pain', 'swollen_legs'],
        'eyeissue': ['redness_of_eyes', 'watering_from_eyes', 'blurred_and_distorted_vision', 'visual_disturbances'],
        'skinissue': ['skin_rash', 'blister', 'itching', 'skin_peeling', 'dischromic _patches'],
        'urinaryissue': ['burning_micturition', 'foul_smell_of urine', 'foul_smell_ofurine', 'yellow_urine', 'polyuria'],
        'vomiting': ['vomiting']
    }
    
    # First check for exact matches in variations
    for key, value in variations.items():
        if key in invalid_symptom.lower():
            matches.extend(value)

    return list(set(matches))

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            idx = symptoms_dict[symptom]
            weight = severity_dict.get(symptom, 1)
            input_vector[idx] = weight

    prediction = svc.predict([input_vector])[0]
    return diseases_list.get(prediction, "Unknown Disease")

# creating routes========================================


@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print(symptoms)
        if not symptoms or symptoms.strip().lower() == "symptoms":
            return render_template('index.html', message="Please enter valid symptoms separated by commas.")

        # Split and clean input symptoms
        user_symptoms = [s.strip().lower() for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

        # Check if all symptoms are valid
        invalid_symptoms = [sym for sym in user_symptoms if sym not in symptoms_dict]
        if invalid_symptoms:
            similar_symptoms = []
            for invalid_sym in invalid_symptoms:
                similar = find_similar_symptoms(invalid_sym)
                if similar:
                    similar_symptoms.extend(similar)
            
            if similar_symptoms:
                message = f"Could not find exact matches for: {', '.join(invalid_symptoms)}.\nDid you mean any of these: {', '.join(set(similar_symptoms))}?"
            else:
                message = f"Could not find matches for: {', '.join(invalid_symptoms)}. Please check the spelling or try different symptoms."
            return render_template('index.html', message=message)

        try:
            predicted_disease = get_predicted_value(user_symptoms)
            if not predicted_disease:
                return render_template('index.html', message="No disease prediction found for the given symptoms. Please try with different symptoms.")

            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                if pd.notna(i):  # Only add non-null values
                    my_precautions.append(i)

            # Process medications, diet, and workout data
            my_medications = []
            if medications and len(medications) > 0:
                med_data = medications[0]
                if isinstance(med_data, str):
                    my_medications = [m.strip() for m in med_data.split(',')]
                else:
                    my_medications = [str(med_data)]

            my_diet = []
            if rec_diet and len(rec_diet) > 0:
                diet_data = rec_diet[0]
                if isinstance(diet_data, str):
                    my_diet = [d.strip() for d in diet_data.split(',')]
                else:
                    my_diet = [str(diet_data)]

            my_workout = []
            if workout and len(workout) > 0:
                workout_data = workout[0]
                if isinstance(workout_data, str):
                    my_workout = [w.strip() for w in workout_data.split(',')]
                else:
                    my_workout = [str(workout_data)]

            # Check if we have any data to display
            if not dis_des and not my_precautions and not my_medications and not my_diet and not my_workout:
                return render_template('index.html', message="Limited information available for this prediction. Please consult a healthcare professional.")

            return render_template('index.html', 
                                predicted_disease=predicted_disease,
                                dis_des=dis_des,
                                my_precautions=my_precautions,
                                medications=my_medications,
                                my_diet=my_diet,
                                workout=my_workout)

        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', message="An error occurred during prediction. Please try again with different symptoms or consult a healthcare professional.")

    return render_template('index.html')


if __name__ == '__main__':

    app.run(debug=True)