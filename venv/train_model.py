import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from matplotlib.ticker import FuncFormatter

# Load Data
df = pd.read_excel(r'C:\Tugas Kuliah\SEMESTER 4\Praktikum AI\Tugas 3\Tugas 3\venv\DATA_RUMAH_SELESAI.xlsx')

X = df[['LB', 'LT']]
X['LOKASI'] = pd.factorize(df['LOKASI'])[0]
y = df['HARGA']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, 'model.pkl')

# Predict
y_pred = model.predict(X_test)

# Evaluasi
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluasi Model:")
print(f"MSE : {mse}")
print(f"MAE : {mae}")
print(f"R² Score : {r2}")

# Buat folder static
os.makedirs('static', exist_ok=True)

# Function Format Harga
def format_rupiah(x, pos):
    if x >= 1_000_000_000:
        return f'Rp {x/1_000_000_000:.1f} M'
    else:
        return f'Rp {x/1_000_000:.0f} jt'

# Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['LB'], y=y_test, color='blue', label='Harga Aktual')
sns.scatterplot(x=X_test['LB'], y=y_pred, color='red', label='Harga Rumah')
plt.xlabel('Luas Bangunan')
plt.ylabel('Harga Rumah')
plt.title('Scatter Plot Luas Bangunan vs Harga Rumah')
plt.legend()
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_rupiah))
plt.grid()
plt.tight_layout()
plt.savefig('static/scatter_plot.png')
plt.close()

# Bar Chart
hasil = pd.DataFrame({'Aktual': y_test.values, 'Prediksi': y_pred})
hasil_sample = hasil.head(20).reset_index(drop=True)

hasil_sample.plot(kind='bar', figsize=(10, 6), color=['blue', 'red'])
plt.title('Perbandingan Harga Aktual dan Harga Rumah')
plt.xlabel('Data Ke-')
plt.ylabel('Harga Rumah')
plt.xticks(rotation=0)
plt.legend(['Harga Aktual', 'Harga Rumah'])
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_rupiah))
plt.grid()
plt.tight_layout()
plt.savefig('static/bar_chart.png')
plt.close()
