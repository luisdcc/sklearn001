"""
Comando de gestión de Django para clasificar colores con KNN.

Este script utiliza Scikit-Learn para entrenar un modelo de clasificación 
de colores basado en la distancia euclidiana o Manhattan con el algoritmo K-Nearest Neighbors (KNN).
Clasifica un color dado en una de las tres categorías: rojo, verde o azul.

Uso:
    python manage.py <nombre_del_comando>

Entradas:
    - RGB de los colores base (rojo, verde, azul).
    - Un color de prueba para clasificar.

Salida:
    - Nombre del color más cercano al de prueba según el modelo entrenado.
"""

import numpy as np
import time
from django.core.management.base import BaseCommand
from sklearn.neighbors import KNeighborsClassifier


class Command(BaseCommand):
    help = 'Clasifica un color dado en una de las tres categorías: rojo, verde o azul'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = 'euclidean'
        # self.metric = 'manhattan'

    def handle(self, *args, **kwargs):
        start_time = time.perf_counter()
        print(f"Inicio del proceso: {start_time:.6f} segundos")

        # Datos de colores (R, G, B) y sus etiquetas
        valores_colores = np.array([
            [255, 0, 0],   # Rojo
            [0, 255, 0],   # Verde
            [0, 0, 255]    # Azul
        ])
        cadena_colores = np.array(["rojo", "verde", "azul"])

        # Crear y entrenar el modelo KNN
        modelo = KNeighborsClassifier(n_neighbors=1, metric=self.metric)
        
        training_start_time = time.perf_counter()
        modelo.fit(valores_colores, cadena_colores)
        training_end_time = time.perf_counter()

        # Color de prueba
        color_prueba = np.array([[200, 0, 0]])

        # Predicción
        prediction_start_time = time.perf_counter()
        resultado = modelo.predict(color_prueba)[0]
        prediction_end_time = time.perf_counter()

        print(f"El color {color_prueba.tolist()[0]} se clasifica como: {resultado}")

        end_time = time.perf_counter()
        print(f"Fin del proceso: {end_time:.6f} segundos")

        # Cálculo de tiempos
        total_time = end_time - start_time
        training_time = training_end_time - training_start_time
        prediction_time = prediction_end_time - prediction_start_time

        print(f"Tiempo total de ejecución: {total_time:.6f} segundos")
        print(f"Tiempo de entrenamiento: {training_time:.6f} segundos")
        print(f"Tiempo de predicción: {prediction_time:.6f} segundos")
