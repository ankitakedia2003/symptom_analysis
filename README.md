
# Symptom Analysis & Medical Recommendation System

This project predicts potential **diseases** based on user-input **symptoms** and provides personalized recommendations including **precautions, medications, diets, workouts**, and a detailed **description**. It leverages machine learning for classification and is deployed as a **Flask web application**.

---

## 🔍 Problem Statement

Diagnosing diseases based on symptoms is often challenging due to overlapping or vague indicators. This project builds a predictive model trained on medical data to classify diseases and offer tailored health advice, empowering users with instant and informative feedback.

---

## 📁 Project Structure

```
symptom_analysis/
├── main.py                    # Flask web application logic
├── svc.pkl                    # Trained Support Vector Classifier model
├── Symptom-severity.csv       # Symptom severity weights
├── description.csv            # Disease descriptions
├── precautions_df.csv         # Recommended precautions
├── medications.csv            # Medication suggestions
├── diets.csv                  # Suggested diets
├── workout_df.csv             # Recommended workouts
├── symtoms_df.csv             # Raw symptoms dataset
├── templates/
│   └── index.html             # HTML interface for symptom input & results
└── static/                    # (Optional) CSS or images
```

---

## 📓 Notebook Highlights: `Medicine Recommendation System.ipynb`

- Data loading & preprocessing from `Training.csv`
- Binary symptom encoding (1/0)
- Model training:
  - SVC (final model), Random Forest, Gradient Boosting, KNN, Naive Bayes
- Accuracy evaluation (all models hit 100% on train/test split)
- Exporting trained `svc.pkl` model
- Dataset shape: 4920 samples × 132 symptoms + 1 diagnosis
- Support functions to fetch recommendations based on disease

---

## 🧠 Features

- Input symptoms via text form (comma-separated)
- Intelligent **disease prediction**
- Fetches and displays:
  - **Description** of the predicted disease
  - **4 Precautions** to take
  - **Recommended Medications**
  - **Diet Suggestions**
  - **Workout Tips**
- Fuzzy matching for similar symptoms in case of typos or general terms
- Severity-based symptom weighting for better predictions

---

## 🎨 Web App

- Flask-based UI with a simple and clean interface
- Accepts symptoms like `headache, nausea, fatigue`
- Returns disease and personalized medical advice
- Error handling for invalid or missing inputs

---

## 🚀 Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/ankitakedia2003/symptom_analysis.git
cd symptom_analysis
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the App

```bash
python main.py
```

Access the app at `http://localhost:5000` in your browser.

---

## 📈 Final Model & Results

- **Model**: Support Vector Classifier (SVC with linear kernel)
- **Accuracy**: 100% on test set (balanced dataset)
- **Output**: Predicted disease index mapped to actual name
- **Evaluation**: Confusion matrix for all models shown in notebook
- **Advantages**:
  - High interpretability
  - Generalizes well to symptoms-based classification

---

## 🌐 Live Demo

- **Render App**: [symptom-analysis-4c5g.onrender.com](https://symptom-analysis-4c5g.onrender.com)

---

## ✨ Built With

- [Scikit-learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Pandas](https://pandas.pydata.org/)
- [HTML/Jinja](https://jinja.palletsprojects.com/)
- [Render](https://render.com/) for deployment
