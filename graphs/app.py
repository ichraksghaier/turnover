from flask import Flask, render_template
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np


app = Flask(__name__)

# Charger le fichier CSV
df = pd.read_csv(r'C:\Users\MSI\Desktop\turnover\turnover_predection\results.csv')

# Mapper les départements
department_mapping = {
    0: "IT", 1: "RandD", 2: "Accounting", 3: "HR", 4: "Management",
    5: "Marketing", 6: "Product Mng", 7: "Sales", 8: "Support", 9: "Technical"
}
df['department'] = df['department'].map(department_mapping)

# Mapper les salaires
salary_mapping = {0: "Low", 1: "Medium", 2: "High"}
df['salary'] = df['salary'].map(salary_mapping)

# Ajouter une colonne pour la prédiction de turnover
df['turnover_prediction'] = df['combined_probability'].apply(lambda x: 'Quitter' if x > 50 else 'Rester')



# Catégoriser les niveaux d'expérience
def categorize_level(years):
    if years < 3:
        return 'Junior'
    elif years <5 :
        return 'Senior'
    else:
        return 'Manager'

df['level'] = df['yearsAtCompany'].apply(categorize_level)

# Mapper le genre : 0 pour femmes, 1 pour hommes
df['gender'] = df['gender'].map({0: 'Femme', 1: 'Homme'})

@app.route('/')
def index():
    # Préparer les données pour le premier graphique
    grouped_df = df.groupby(['department', 'turnover_prediction']).size().reset_index(name='count')
    total_per_department = grouped_df.groupby('department')['count'].transform('sum')
    grouped_df['percentage'] = (grouped_df['count'] / total_per_department * 100).round(2)

    # Graphique 1 : Répartition des employés par turnover et département
    fig1 = px.bar(
        grouped_df,
        x='department',
        y='count',
        color='turnover_prediction',
        text='percentage',
        title="Répartition des employés par département ",
        barmode='group',
        color_discrete_map={'Quitter': '#E74C3C', 'Rester': '#2ECC71'}
    )

    fig1.update_traces(
        texttemplate='%{text}%', 
        textposition='outside', 
        textfont=dict(size=14, color="black")
    )
    fig1.update_layout(
        width=800,  # Largeur augmentée
        height=500,  # Ajustez la hauteur
        title_font=dict(size=18, family="Arial, sans-serif", color='#333'),
        xaxis_title="Département",
        yaxis_title="Nombre d'employés",
        legend_title="Turnover",
        plot_bgcolor='rgba(0, 0, 0, 0)',
        
    
)
    fig1_html = fig1.to_html(full_html=False)

    # Graphique 2 : Répartition des niveaux par turnover (barres empilées)
    level_turnover = df.groupby(['level', 'turnover_prediction']).size().reset_index(name='count')
    total_per_level = level_turnover.groupby('level')['count'].transform('sum')
    level_turnover['percentage'] = (level_turnover['count'] / total_per_level * 100).round(2)

    fig2 = px.bar(
        level_turnover,
        x='level',
        y='count',
        color='turnover_prediction',
        text='percentage',
        title="Répartition des employés par niveau ",
        barmode='stack',
        color_discrete_map={'Quitter': '#E74C3C', 'Rester': '#2ECC71'}
    )

    fig2.update_traces(
        texttemplate='%{text}%', 
        textposition='inside', 
        textfont=dict(size=14, color="white")
    )
    fig2.update_layout(
        title_font=dict(size=18, family="Arial, sans-serif", color='#333'),
        xaxis_title="Niveau d'expérience",
        yaxis_title="Nombre d'employés",
        legend_title="Turnover",
        bargap=0.3,
        title_x=0.5,
        uniformtext_minsize=12,
        xaxis=dict(tickvals=['Junior', 'Senior', 'Manager'], ticktext=['Junior', 'Senior', 'Manager']),
        plot_bgcolor='rgba(0, 0, 0, 0)',
    
    )
    fig2_html = fig2.to_html(full_html=False)

    # Graphique 3 : Répartition des salaires par turnover
    fig3 = px.bar(
        df,
        y='salary',
        color='turnover_prediction',
        title="Répartition des salaires ",
        orientation='h',
        barmode='group',
        color_discrete_map={'Quitter': '#E74C3C', 'Rester': '#2ECC71'}
    )

    fig3.update_layout(
        xaxis_title="Nombre d'employés",
        yaxis_title="Niveaux de salaire",
        legend_title="Statut de Turnover",
        bargap=0.1,
        barmode='stack',
        title_font=dict(size=18, family="Arial, sans-serif", color='#333'),
        plot_bgcolor='rgba(0, 0, 0, 0)',
    )
    fig3_html = fig3.to_html(full_html=False)

    # Graphique 4 : Répartition des genres par turnover
    gender_turnover = df.groupby(['gender', 'turnover_prediction']).size().reset_index(name='count')
    total_per_gender = gender_turnover.groupby('gender')['count'].transform('sum')
    gender_turnover['percentage'] = (gender_turnover['count'] / total_per_gender * 100).round(2)

    fig4 = px.bar(
        gender_turnover,
        x='gender',
        y='count',
        color='turnover_prediction',
        text='percentage',
        title="Répartition des genres ",
        barmode='group',
        color_discrete_map={'Quitter': '#E74C3C', 'Rester': '#2ECC71'}
    )

    fig4.update_traces(
        texttemplate='%{text}%', 
        textposition='outside', 
        textfont=dict(size=14, color="black")
    )
    fig4.update_layout(
        title_font=dict(size=18, family="Arial, sans-serif", color='#333'),
        xaxis_title="Genre",
        yaxis_title="Nombre d'employés",
        legend_title="Turnover",
        bargap=0.3,
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        plot_bgcolor='rgba(0, 0, 0, 0)',
    )
    fig4_html = fig4.to_html(full_html=False)

    # Graphique 5 : Répartition des années d'ancienneté par turnover
    fig5 = go.Figure()

    # Courbe de densité pour les employés qui vont quitter (turnover == 'Quitter')
    kde_quitter = gaussian_kde(df.loc[df['turnover_prediction'] == 'Quitter', 'evaluation'])
    x_vals_quitter = np.linspace(df['evaluation'].min(), df['evaluation'].max(), 1000)
    fig5.add_trace(go.Scatter(
        x=x_vals_quitter,
        y=kde_quitter(x_vals_quitter),
        mode='lines',
        name='Quitter',
        line=dict(color='#E74C3C'),
        fill='tozeroy',  # Remplir sous la courbe
    ))

    # Courbe de densité pour les employés qui restent (turnover == 'Rester')
    kde_rester = gaussian_kde(df.loc[df['turnover_prediction'] == 'Rester', 'evaluation'])
    x_vals_rester = np.linspace(df['evaluation'].min(), df['evaluation'].max(), 1000)
    fig5.add_trace(go.Scatter(
        x=x_vals_rester,
        y=kde_rester(x_vals_rester),
        mode='lines',
        name='Rester',
        line=dict(color='#2ECC71'),
        fill='tozeroy',  # Remplir sous la courbe
    ))

    # Mise à jour de la mise en page
    fig5.update_layout(
        title="Distribution des évaluations des employés ",
        xaxis_title="Evaluation",
        yaxis_title="Densité",
        legend_title="Turnover",
        title_font=dict(size=18, family="Arial, sans-serif", color='#333'),
    )

    fig5_html = fig5.to_html(full_html=False)


    # Graphique 6 : Distribution des heures mensuelles moyennes par turnover
    fig6 = go.Figure()

    # Courbe de densité pour les employés qui vont quitter (turnover == 'Quitter')
    kde_quitter_hours = gaussian_kde(df.loc[df['turnover_prediction'] == 'Quitter', 'averageMonthlyHours'])
    x_vals_quitter_hours = np.linspace(df['averageMonthlyHours'].min(), df['averageMonthlyHours'].max(), 1000)
    fig6.add_trace(go.Scatter(
        x=x_vals_quitter_hours,
        y=kde_quitter_hours(x_vals_quitter_hours),
        mode='lines',
        name='Quitter',
        line=dict(color='#E74C3C'),
        fill='tozeroy',  # Remplir sous la courbe
    ))

    # Courbe de densité pour les employés qui restent (turnover == 'Rester')
    kde_rester_hours = gaussian_kde(df.loc[df['turnover_prediction'] == 'Rester', 'averageMonthlyHours'])
    x_vals_rester_hours = np.linspace(df['averageMonthlyHours'].min(), df['averageMonthlyHours'].max(), 1000)
    fig6.add_trace(go.Scatter(
        x=x_vals_rester_hours,
        y=kde_rester_hours(x_vals_rester_hours),
        mode='lines',
        name='Rester',
        line=dict(color='#2ECC71'),
        fill='tozeroy',  # Remplir sous la courbe
    ))

    # Mise à jour de la mise en page
    fig6.update_layout(
        title="Distribution des heures mensuelles moyennes",
        xaxis_title="Heures mensuelles moyennes",
        yaxis_title="Densité",
        legend_title="Turnover",
        title_font=dict(size=18, family="Arial, sans-serif", color='#333'),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        height=450,  # Hauteur similaire à fig3, fig4, fig5
        width=400    # Largeur similaire
    )

    fig6_html = fig6.to_html(full_html=False)

    fig7 = go.Figure()
    fig7 = px.pie(df, values=df['turnover_prediction'].value_counts().values, names=df['turnover_prediction'].value_counts().index,
                title="Prédiction de Turnover : Quitter ou Rester",
                color_discrete_sequence=px.colors.sequential.Blues)

    # Ajuster la taille pour qu'elle soit similaire à la figure 5
    fig7.update_layout(
        width=900,
        height=600,
        font=dict(size=16),
        title_font=dict(size=20),
        legend=dict(font=dict(size=14))
    )

    # Exporter le graphique en HTML
    fig7_html = fig7.to_html(full_html=False)

    # Calcul des KPIs pour le turnover
    total_employees = len(df)
    num_quitter = len(df[df['turnover_prediction'] == 'Quitter'])
    num_rester = len(df[df['turnover_prediction'] == 'Rester'])
    quit_rate = (num_quitter / total_employees) * 100
    stay_rate = (num_rester / total_employees) * 100

    # Créer une figure pour afficher les KPIs
    fig_kpi = go.Figure()

    # Ajouter des annotations pour les KPIs
    fig_kpi.add_trace(go.Scatter(
        x=[0.5], y=[0.9],
        text=f"Total Employés: {total_employees}",
        mode="text",
        showlegend=False,
        textfont=dict(size=20, color="black")
    ))

    fig_kpi.add_trace(go.Scatter(
        x=[0.5], y=[0.75],
        text=f"Quitter: {num_quitter} ({quit_rate:.2f}%)",
        mode="text",
        showlegend=False,
        textfont=dict(size=20, color="red")
    ))

    fig_kpi.add_trace(go.Scatter(
        x=[0.5], y=[0.6],
        text=f"Rester: {num_rester} ({stay_rate:.2f}%)",
        mode="text",
        showlegend=False,
        textfont=dict(size=20, color="green")
    ))

    # Ajout d'autres KPIs (exemples : moyenne des années d'ancienneté et évaluation)
    avg_tenure = df['yearsAtCompany'].mean()
    avg_evaluation = df['evaluation'].mean()

    fig_kpi.add_trace(go.Scatter(
        x=[0.5], y=[0.45],
        text=f"Ancienneté moyenne: {avg_tenure:.2f} ans",
        mode="text",
        showlegend=False,
        textfont=dict(size=17, color="blue"),
        textposition="middle center"
    ))

    fig_kpi.add_trace(go.Scatter(
        x=[0.5], y=[0.3],
        text=f"Evaluation moyenne: {avg_evaluation:.2f}",
        mode="text",
        showlegend=False,
        textfont=dict(size=18, color="purple")
    ))

    # Mise à jour de la mise en page pour la figure des KPIs
    fig_kpi.update_layout(
        xaxis=dict(
            showticklabels=False  # Masque les labels de l'axe X
        ),
        yaxis=dict(
            showticklabels=False  # Masque les labels de l'axe Y
        ),
        title_font=dict(size=20, family="Arial, sans-serif", color='#333',weight='normal'),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        title="Kpis sur le  turnover",
        title_x=0.5,

)

    # Exporter le graphique en HTML
    fig_kpi_html = fig_kpi.to_html(full_html=False)


    # Retourner la page HTML avec les graphiques
    return render_template("index.html", fig1_html=fig1_html, fig2_html=fig2_html, fig3_html=fig3_html, fig4_html=fig4_html, fig5_html=fig5_html,fig6_html=fig6_html,fig_kpi_html=fig_kpi_html)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
