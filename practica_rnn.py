import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=100, precision=4, sci_mode=False)


# 1. Generación de Datos Sintéticos (Serie Temporal de una Onda Senoidal)
def generar_datos(n_puntos=500, seq_len=10):
    t = np.linspace(0, 50, n_puntos)
    serie = np.sin(t)
    X, y = [], []
    for i in range(len(serie) - seq_len):
        X.append(serie[i:i+seq_len])
        y.append(serie[i+seq_len])
    # Convertir a tensores de PyTorch con dimensiones tridimensionales (Batch, Seq_Len, Features)
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
    return X, y

# 2. Definición de la Arquitectura RNN Simple
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # out: [batch_size, seq_len, hidden_size]
        out, _ = self.rnn(x)
        # Extraemos exclusivamente el último estado temporal (Many-to-One)
        out_final = out[:, -1, :]
        pred = self.fc(out_final)
        return pred

# 3. Inicialización de Datos
SEQ_LEN = 50
X, y = generar_datos(n_puntos=600, seq_len=SEQ_LEN)

# --- ACTIVIDAD 1: Inspección de dimensiones ---
print("=== INSPECCIÓN DE TENSORES 3D ===")
print(f"X.shape = {X.shape}")
print(f"y.shape = {y.shape}")
print(f"Número de muestras (ventanas): {X.shape[0]}")
print(f"Longitud de secuencia:          {X.shape[1]}")
print(f"Features por paso de tiempo:    {X.shape[2]}")
print(f"\nPrimera secuencia X[0] (50 valores):")
print(X[0].flatten())
print(f"\nPrimera etiqueta y[0]: {y[0].item():.4f}")
print("=================================\n")

# Inicialización del modelo, pérdida y optimización
model = SimpleRNNModel(input_size=1, hidden_size=16, output_size=1)
print(f"[hidden=16] Parámetros entrenables: {sum(p.numel() for p in model.parameters())}")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)



# 4. Bucle de Entrenamiento (con historial de pérdida)
epochs = 50
loss_history = []
for epoch in range(epochs):
    model.train()
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

print("¡Entrenamiento de la RNN Simple finalizado con éxito!")

# 5. Evaluación y visualización
model.eval()
with torch.no_grad():
    preds = model(X).numpy().flatten()
y_real = y.numpy().flatten()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history, color='tab:red')
plt.title('Curva de Pérdida — RNN Simple')
plt.xlabel('Época')
plt.ylabel('MSE Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(y_real[:200], label='Real', alpha=0.8)
plt.plot(preds[:200], label='Predicho', alpha=0.8)
plt.title('Predicción vs Real (primeros 200 puntos)')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()