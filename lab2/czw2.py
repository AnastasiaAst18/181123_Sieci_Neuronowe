import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#wczytanie danych
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

#analiza danych – liczba wierszy, liczba cech, podział na klasy
num_rows, num_features = X.shape
classes, counts = np.unique(y, return_counts=True)

#normalizacja
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_n = (X - X_min)/(X_max - X_min)
#print(X_n[:5])

#utworzenie neuronów
np.random.seed(42)
indices = np.arange(len(X_n))
np.random.shuffle(indices)
X_shuffled = X_n[indices]
y_shuffled = y[indices]
weights = np.random.rand(3, 4)
#print(weights)

#implementacja odległości i znalezienie neuronu zwycięskiego
def find_n(x, weights):
    distance = np.sqrt(np.sum((weights - x) ** 2, axis=1))
    n_index = np.argmin(distance)
    return n_index

#implementacja aktualizacji wag z dynamicznym współczynnikiem uczenia
initial_learning_rate = 0.1
epochs = 50
for epoch in range(epochs):
  learning_rate = initial_learning_rate * (1 - epoch / epochs) #modyfikacja współczynnika uczenia
  if epoch % 10 == 0: #wyświetlenie wartości współczynnika co 10 epok
    print(f"Epoka {epoch}, współczynnik uczenia: {learning_rate:.4f}")
  for x in X_shuffled:
    n_index = find_n(x, weights)
    weights[n_index] = weights[n_index] + learning_rate * (x - weights[n_index])
print("Końcowe wagi 3 neuronów: ", weights)

#przypisanie przykładów do zwycięskich neuronów
predicted = []
for x in X_shuffled:
  n_index = find_n(x, weights)
  predicted.append(n_index)
predicted = np.array(predicted)

#przygotowanie macierzy (tabeli) rozkładu przykładów w klasach
conf_matrix = np.zeros((3, 3), dtype=int)
for neuron_idx, true_class in zip(predicted, y_shuffled):
  conf_matrix[neuron_idx, true_class] += 1

#formatowanie tabeli
df_matrix = pd.DataFrame(
    conf_matrix,
    index=['N1', 'N2', 'N3'],
    columns=['Setosa', 'Versicolor', 'Virginica'],
)
print(df_matrix)