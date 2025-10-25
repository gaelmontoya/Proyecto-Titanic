# Crea este nuevo archivo: predictor/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Cuando alguien visite la raíz del sitio, llama a la función 'predict_page'
    path('', views.predict_page, name='predict_page'),
]