import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

print("Iniciando el entrenamiento del modelo (v2 con 'Embarked')...")

# 1. Cargar Datos
try:
    df = pd.read_csv("Titanic-Dataset.csv")
except FileNotFoundError:
    print("Error: No se encontró 'Titanic-Dataset.csv'.")
    exit()

# 2. Definir Features (X) y Target (y)
# ¡AÑADIMOS 'Embarked' A LAS FEATURES!
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'

X = df[features]
y = df[target]

# 3. Preprocesamiento

# Definir columnas numéricas y categóricas
numeric_features = ['Age', 'Fare']
# ¡AÑADIMOS 'Embarked' A LAS CATEGÓRICAS!
categorical_features = ['Pclass', 'Sex', 'Embarked']

# --- Pipelines de Transformación ---

# Pipeline numérico (sin cambios)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline categórico (ACTUALIZADO)
# Ahora tiene 2 pasos:
# 1. Imputer: Rellena datos faltantes ('NaN') con el valor más frecuente.
# 2. OneHotEncoder: Convierte las categorías en números.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar ambos transformadores en el Preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Crear el Pipeline del Modelo Final (sin cambios)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# 5. Entrenar el Modelo
model_pipeline.fit(X, y)

# 6. Guardar el Modelo
joblib.dump(model_pipeline, 'titanic_model.joblib')

print("¡Éxito! Modelo (v2) entrenado y guardado como 'titanic_model.joblib'")