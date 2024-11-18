from flask import Flask, request, jsonify, render_template
import pandas as pd  # Importer pandas
from functions import calculate_turnover_risk, combine_probabilities
import joblib

# Charger le modèle de machine learning
model = joblib.load('rf_model.pkl')

# Initialiser une DataFrame globale pour stocker les résultats
results_df = pd.DataFrame(columns=[
    'satisfaction', 'evaluation', 'projectCount', 'averageMonthlyHours', 
    'yearsAtCompany', 'workAccident', 'promotion', 'department', 
    'salary', 'gender', 'combined_probability'
])

def apply_model(data, model):
    features = [
        data['satisfaction'], data['evaluation'], data['projectCount'],
        data['averageMonthlyHours'], data['yearsAtCompany'], data['workAccident'],
        data['promotion'], data['department'], data['salary'], data['gender']
    ]
    features = [features]  
    model_probability = model.predict_proba(features)[0][1] * 100  
    return model_probability

# Création de l'application Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global results_df  # Utiliser la DataFrame globale

    # Récupérer les données de la requête
    data = request.get_json()
    print(f"Received data: {data}")  # Afficher les données reçues pour débogage

    # Appliquer le modèle pour obtenir la probabilité
    model_probability = apply_model(data, model)

    # Calculer le risque ajusté
    adjusted_probability = calculate_turnover_risk(
        data['satisfaction'], data['evaluation'], data['projectCount'],
        data['averageMonthlyHours'], data['yearsAtCompany'], data['workAccident'],
        data['promotion'], data['department'], data['salary'], data['gender']
    )

    # Combiner les deux probabilités
    combined_probability = combine_probabilities(model_probability, adjusted_probability)

    # Ajouter les résultats à la DataFrame
    new_row = pd.DataFrame([{
        'satisfaction': data['satisfaction'],
        'evaluation': data['evaluation'],
        'projectCount': data['projectCount'],
        'averageMonthlyHours': data['averageMonthlyHours'],
        'yearsAtCompany': data['yearsAtCompany'],
        'workAccident': data['workAccident'],
        'promotion': data['promotion'],
        'department': data['department'],
        'salary': data['salary'],
        'gender': data['gender'],
        'combined_probability': combined_probability
    }])
    # Concaténer sans provoquer d'avertissements
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Sauvegarder la DataFrame dans un fichier CSV
    results_df.to_csv('results.csv', index=False)

    # Retourner les résultats sous forme JSON
    return jsonify({
        "model_probability": model_probability,
        "adjusted_probability": adjusted_probability,
        "combined_probability": combined_probability
    })

# Exécuter l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
