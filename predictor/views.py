from django.shortcuts import render
import joblib
import pandas as pd
import os

# 1. Cargar el modelo (esto no cambia)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'titanic_model.joblib')
model = joblib.load(MODEL_PATH)


def predict_page(request):
    """
    Renderiza la página del formulario y maneja el cálculo de la predicción.
    """
    context = {} 
    
    if request.method == 'POST':
        try:
            # 1. Obtener datos del formulario
            pclass = int(request.POST.get('pclass'))
            sex = request.POST.get('sex')
            age = float(request.POST.get('age'))
            fare = float(request.POST.get('fare'))
            # --- ¡NUEVO CAMPO! ---
            embarked = request.POST.get('embarked') # 'S', 'C' o 'Q'

            # 2. Crear un DataFrame (con la nueva columna 'Embarked')
            input_data = pd.DataFrame({
                'Pclass': [pclass],
                'Sex': [sex],
                'Age': [age],
                'Fare': [fare],
                'Embarked': [embarked] # ¡Añadido!
            })
            
            # 3. Predecir la probabilidad (esto no cambia)
            probability = model.predict_proba(input_data)
            survival_prob = probability[0][1] 
            
            # 4. Formatear el resultado (esto no cambia)
            context['result'] = f"{survival_prob * 100:.2f}%"
            context['inputs'] = request.POST 

        except Exception as e:
            context['error'] = f"Error al procesar los datos: {e}"

    return render(request, 'predictor/predict.html', context)