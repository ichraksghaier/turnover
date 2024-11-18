

# Fonction pour calculer le risque de turnover ajusté
def calculate_turnover_risk(satisfaction, evaluation, project_count, average_monthly_hours, 
                            years_at_company, work_accident, promotion, department, salary, gender):
    # Assurer que department, salary et gender sont des entiers
    department = int(department)
    salary = int(salary)
    gender = int(gender)

    # Initialisation du risque de turnover
    turnover_risk = 0.1  # Risque de base

    # Mapping pour department et salary
    department_mapping = {
        0: "IT", 1: "RandD", 2: "accounting", 3: "hr", 4: "management",
        5: "marketing", 6: "product_mng", 7: "sales", 8: "support", 9: "technical"
    }
    salary_mapping = {0: "High", 1: "Low", 2: "Medium"}

    # Conversion de department et salary en valeurs textuelles
    department_text = department_mapping.get(department, "unknown")
    salary_text = salary_mapping.get(salary, "Medium")  # Par défaut "Medium" pour un salaire inconnu

    # Ajustement basé sur la satisfaction
    turnover_risk += (1 - satisfaction) * 0.3  # Plus la satisfaction est faible, plus le risque augmente

    # Ajustement basé sur le genre
    turnover_risk += 0.1 if gender == 1 else -0.02  # Les hommes ont un léger risque plus élevé

    # Ajustements basés sur le department
    if department_text in ['management', 'hr', 'technical', 'accounting']:
        turnover_risk -= 0.05
    elif department_text in ['sales', 'IT', 'marketing', 'RandD', 'product_mng', 'support']:
        turnover_risk += 0.05

    # Ajustement basé sur les années d'expérience
    experience_factor = 0
    if years_at_company < 3:
        experience_factor = 0.1
    elif 3 <= years_at_company <= 5:
        experience_factor = 0.2
    elif years_at_company > 5:
        experience_factor = -0.1

    # Ajustement basé sur le salaire
    salary_factor = 0
    if salary_text == 'Low':
        salary_factor = 0.15
        if years_at_company > 3:
            experience_factor += 0.2
    elif salary_text == 'Medium':
        salary_factor = 0.1
        if years_at_company > 5:
            experience_factor += 0.1
    elif salary_text == 'High':
        salary_factor = -0.2

    # Appliquer les facteurs combinés
    turnover_risk += experience_factor + salary_factor

    # Ajustement basé sur la promotion
    if not promotion:
        if salary_text == 'Low':
            turnover_risk += 0.2
        elif salary_text == 'Medium':
            turnover_risk += 0.1
        elif salary_text == 'High':
            turnover_risk += 0.05

    # Accident de travail
    if work_accident:
        turnover_risk += 0.1

    # Ajustement basé sur les heures de travail mensuelles
    if average_monthly_hours < 120 or average_monthly_hours > 230:
        turnover_risk += 0.05

    # Ajustement basé sur l'évaluation de performance
    turnover_risk -= evaluation * 0.2

    # Nombre de projets
    if project_count > 5:
        turnover_risk += 0.05
        if salary_text == 'Low' and not promotion:
            turnover_risk += 0.1  # Double pénalité si le salaire est bas et pas de promotion
    elif project_count < 3:
        turnover_risk -= 0.05

    # Limiter le risque de turnover entre 0% et 100%
    turnover_risk_percentage = max(0, min(turnover_risk * 100, 100))
    return turnover_risk_percentage

# Fonction pour combiner les probabilités du modèle et du risque ajusté
def combine_probabilities(probability1, probability2, weight1=0.1, weight2=0.9):
    return weight1 * probability1 + weight2 * probability2
