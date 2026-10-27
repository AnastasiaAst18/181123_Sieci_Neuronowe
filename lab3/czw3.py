import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# przygotowanie wzorcowych obrazów. tworzymy 3 przykładowe wzorce
p_1 = np.array([
    -1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
])

p_2 = np.array([
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
])

p_3 = np.array([
    -1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
])

p = [p_1, p_2, p_3]

#funkcja dodająca szum do obrazu
def add_noise(p, noise_level):
  noisy = p.copy()
  num_pixels = len(p)
  num_noisy = int(num_pixels * noise_level)
  #wybieramy losowe indeksy do zmiana szumu
  ind = np.random.choice(num_pixels, num_noisy, replace=False)
  for idx in ind:
    noisy[idx] *= -1
  return noisy

#generujemy zaszumione wersje
noise_level = 0.2
noisy_p = [add_noise(a, noise_level) for a in p]

#implementacja sieci Hopfielda i reguły Hebba
class HopfieldNetwork:
  def __init__(self, n_neurons):
    self.n_neurons = n_neurons
    self.weights = np.zeros((n_neurons, n_neurons))

  def train(self, p):
    for a in p:
      self.weights += np.outer(a, a)
    np.fill_diagonal(self.weights, 0)

  def recall(self, pt, steps=5):
    s = pt.copy()
    for _ in range(steps):
      for i in range(self.n_neurons):
        h = np.dot(self.weights[i], s)
        if h > 0:
          s[i] = 1
        elif h < 0:
          s[i] = -1
    return s

#inicjalizacja i uczenie sieci
hopfield = HopfieldNetwork(n_neurons=64)
hopfield.train(p)

#sprawdzenie warunków
#czy macierz wag jest symetryczna?(wij == wji)
is_symmetric = np.allclose(hopfield.weights, hopfield.weights.T)
print(f"Czy macierz wag jest symetryczna? {is_symmetric}")

#czy na przekątnej znajdują się zera? (wii == 0)
has_zeros_on_diag = np.all(np.diag(hopfield.weights) == 0)
print(f"Czy na przekątnej znajdują się zera? {has_zeros_on_diag}")

#test odtwarzania zaszumionego wzorca
restored_p = hopfield.recall(noisy_p[0], steps=5)

#sprawdzenie czy odtworzono oryginał
is_restored = np.array_equal(restored_p, p[0])
print(f"Czy zaszumiony wzorzec został poprawnie odtworzony? {is_restored}")

#wizualizacja za pomocą matplotlib (wzorzec, zaszumiony, odtworziony)
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i in range(3):
  axes[i, 0].imshow(p[i].reshape(8, 8), cmap='gray')
  axes[i, 0].set_title(f"Oryginał {i+1}")

  axes[i, 1].imshow(noisy_p[i].reshape(8, 8), cmap='gray')
  axes[i, 1].set_title(f"Zaszumiony ({int(noise_level*100)}%)")

  restored = hopfield.recall(noisy_p[i], steps=5)
  axes[i, 2].imshow(restored.reshape(8, 8), cmap='gray')
  axes[i, 2].set_title(f"Odtworzony {i+1}")

plt.tight_layout()
plt.show()