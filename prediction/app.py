from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from io import BytesIO
from functions import calculate_turnover_risk, combine_probabilities

# Charger le modèle de machine learning
model = joblib.load('rf_model.pkl')

# Vérifiez si le fichier CSV existe et chargez-le si nécessaire
if os.path.exists('results.csv'):
    results_df = pd.read_csv('results.csv')  # Charger les anciennes données
else:
    # Si le fichier n'existe pas, créez un DataFrame vide
    results_df = pd.DataFrame(columns=[  
        'satisfaction', 'evaluation', 'projectCount', 'averageMonthlyHours', 
        'yearsAtCompany', 'workAccident', 'promotion', 'department', 
        'salary', 'gender', 'combined_probability'
    ])

# Fonction pour appliquer le modèle sur les nouvelles données
def apply_model(data, model):
    features = [
        data['satisfaction'], data['evaluation'], data['projectCount'],
        data['averageMonthlyHours'], data['yearsAtCompany'], data['workAccident'],
        data['promotion'], data['department'], data['salary'], data['gender']
    ]
    features = [features]  
    model_probability = model.predict_proba(features)[0][1] * 100  
    return model_probability

# Mapping des départements et salaires
department_mapping = {
    0: "IT", 1: "RandD", 2: "Accounting", 3: "HR", 4: "Management",
    5: "Marketing", 6: "Product Mng", 7: "Sales", 8: "Support", 9: "Technical"
}

salary_mapping = {0: "Low", 1: "Medium", 2: "High"}

# Fonction pour appliquer les mappings sur les données
def apply_mappings(dataframe):
    dataframe['department'] = dataframe['department'].map(department_mapping)
    dataframe['salary'] = dataframe['salary'].map(salary_mapping)

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

    # Concaténer les nouvelles lignes avec les anciennes
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Sauvegarder la DataFrame mise à jour dans le fichier CSV
    results_df.to_csv('results.csv', index=False)

    # Retourner les résultats sous forme JSON avec des graphiques
    return jsonify({
        "model_probability": model_probability,
        "adjusted_probability": adjusted_probability,
        "combined_probability": combined_probability,
        "graph_url": '/graphs'
    })

# Route pour afficher les graphiques
@app.route('/graphs')
def show_graphs():
    global results_df

    # Histogramme de la probabilité combinée
    plt.figure(figsize=(8, 6))
    plt.hist(results_df['combined_probability'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution de la probabilité combinée de turnover')
    plt.xlabel('Probabilité (%)')
    plt.ylabel('Fréquence')
    plt.tight_layout()

    # Sauvegarder le graphique dans un objet BytesIO
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Retourner le graphique comme image
    return send_file(img, mimetype='image/png')

# Exécuter l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
