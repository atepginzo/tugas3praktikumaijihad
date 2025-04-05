from flask import Flask, render_template, request
import os
import joblib
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

# Load data untuk mendapatkan lokasi
df = pd.read_excel(r'C:\Tugas Kuliah\SEMESTER 4\Praktikum AI\Tugas 3\Tugas 3\venv\DATA_RUMAH_SELESAI.xlsx')
lokasi_unik = df['LOKASI'].unique()

# Route Halaman Utama
@app.route('/')
def index():
    return render_template('index.html', lokasi_list=lokasi_unik)

# Route untuk Prediksi Manual
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        # Ambil input dari form
        lb = float(request.form['luas_bangunan'])
        lt = float(request.form['luas_tanah'])
        lokasi = request.form['lokasi']

        # Prediksi harga rumah
        data_input = pd.DataFrame([[lb, lt]], columns=['LB', 'LT'])
        lokasi_int = df[df['LOKASI'] == lokasi].index[0]  # Ambil index lokasi yang sesuai
        data_input['LOKASI'] = lokasi_int  # Assign lokasi yang dipilih ke input
        prediksi_harga = model.predict(data_input)[0]

        # Format harga ke Rupiah
        def format_rupiah(x):
            if x >= 1_000_000_000:
                return f'Rp {x / 1_000_000_000:.1f} M'
            else:
                return f'Rp {x / 1_000_000:.0f} jt'

        harga_rupiah = format_rupiah(prediksi_harga)

        return render_template('index.html', prediksi=harga_rupiah, lokasi_list=lokasi_unik)

    except Exception as e:
        return f"Terjadi kesalahan: {e}"

# Jalankan Aplikasi
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
    
