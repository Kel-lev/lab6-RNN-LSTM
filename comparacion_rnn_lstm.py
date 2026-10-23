import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

# 1. Datos
def generar_datos(n_puntos=600, seq_len=50):
    t = np.linspace(0, 50, n_puntos)
    serie = np.sin(t)
    X, y = [], []
    for i in range(len(serie) - seq_len):
        X.append(serie[i:i+seq_len])
        y.append(serie[i+seq_len])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
    return X, y

# 2. Arquitecturas
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.fc(out[:, -1, :])

# 3. Función de entrenamiento reutilizable
def entrenar(modelo, X, y, epochs=50, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        modelo.train()
        pred = modelo(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# 4. Configuración del experimento
SEQ_LEN = 50
HIDDEN = 16
EPOCHS = 50
X, y = generar_datos(n_puntos=600, seq_len=SEQ_LEN)

print(f"X.shape = {X.shape}   |   SEQ_LEN = {SEQ_LEN}   |   HIDDEN = {HIDDEN}\n")

# 5. Entrenar ambos modelos
torch.manual_seed(42)
rnn = SimpleRNNModel(1, HIDDEN, 1)
print(f"RNN  params: {sum(p.numel() for p in rnn.parameters())}")
loss_rnn = entrenar(rnn, X, y, EPOCHS)

torch.manual_seed(42)
lstm = LSTMModel(1, HIDDEN, 1)
print(f"LSTM params: {sum(p.numel() for p in lstm.parameters())}")
loss_lstm = entrenar(lstm, X, y, EPOCHS)

print(f"\nLoss final RNN : {loss_rnn[-1]:.6f}")
print(f"Loss final LSTM: {loss_lstm[-1]:.6f}")

# 6. Visualización comparativa
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(loss_rnn, label='RNN Simple', color='tab:red', linewidth=2)
axes[0].plot(loss_lstm, label='LSTM', color='tab:blue', linewidth=2)
axes[0].set_title(f'Convergencia — RNN vs LSTM (SEQ_LEN={SEQ_LEN}, hidden={HIDDEN})')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('MSE Loss')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

rnn.eval(); lstm.eval()
with torch.no_grad():
    pred_rnn = rnn(X).numpy().flatten()
    pred_lstm = lstm(X).numpy().flatten()
y_real = y.numpy().flatten()

n = 200
axes[1].plot(y_real[:n], label='Real', color='black', alpha=0.6, linewidth=2)
axes[1].plot(pred_rnn[:n], label='Pred RNN', color='tab:red', alpha=0.85)
axes[1].plot(pred_lstm[:n], label='Pred LSTM', color='tab:blue', alpha=0.85)
axes[1].set_title('Predicción vs Real (primeros 200 puntos)')
axes[1].set_xlabel('Tiempo')
axes[1].set_ylabel('Amplitud')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparacion_rnn_lstm.png', dpi=120)   # guarda la imagen
plt.show()