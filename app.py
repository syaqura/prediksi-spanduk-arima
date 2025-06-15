import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Prediksi Pesanan", layout="centered")
st.title("📦 Prediksi Pesanan")
st.write("Selamat datang di aplikasi prediksi pesanan spanduk berbasis ARIMA.")

# Upload data CSV
uploaded_file = st.file_uploader("📂 Upload file CSV kamu", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';')
    except:
        st.error("❌ Gagal membaca file. Pastikan format CSV dan separator benar.")
    else:
        if 'bulan' in df.columns and 'spanduk' in df.columns:
            try:
                df['bulan'] = pd.to_datetime(df['bulan'], format='%d/%m/%Y')
                df.set_index('bulan', inplace=True)
                df = df.asfreq('MS')
            except Exception as e:
                st.error(f"❌ Format tanggal tidak valid: {e}")
            else:
                st.success("✅ Data berhasil dimuat")
                st.dataframe(df.tail())

                st.subheader("📊 Grafik Jumlah Pesanan")
                st.line_chart(df['spanduk'])

                if st.checkbox("🔬 Tampilkan Uji Stasioneritas dan ACF/PACF"):
                    st.subheader("📉 Uji Stasioneritas (ADF Test)")
                    adf_result = adfuller(df['spanduk'].dropna())
                    st.write(f"ADF Statistic: {adf_result[0]}")
                    st.write(f"p-value: {adf_result[1]:.10f}")
                    if adf_result[1] <= 0.05:
                        st.success("✅ Data bersifat stasioner (p-value <= 0.05)")
                    else:
                        st.warning("⚠️ Data tidak stasioner (p-value > 0.05), disarankan lakukan differencing (d ≥ 1)")

                    st.subheader("📈 Grafik ACF")
                    fig_acf, ax_acf = plt.subplots()
                    plot_acf(df['spanduk'].dropna(), ax=ax_acf, lags=20)
                    st.pyplot(fig_acf)

                    st.subheader("📈 Grafik PACF")
                    fig_pacf, ax_pacf = plt.subplots()
                    plot_pacf(df['spanduk'].dropna(), ax=ax_pacf, lags=20)
                    st.pyplot(fig_pacf)

                st.subheader("⚙️ Parameter ARIMA")
                p = st.number_input("Masukkan nilai p", min_value=0, max_value=10, value=1)
                d = st.number_input("Masukkan nilai d", min_value=0, max_value=2, value=1)
                q = st.number_input("Masukkan nilai q", min_value=0, max_value=10, value=1)

                show_accuracy = st.checkbox("📏 Hitung akurasi model dari 6 bulan terakhir")

                if st.button("🔮 Jalankan Prediksi"):
                    n_test = 6
                    if len(df) > n_test:
                        train_data = df['spanduk'][:-n_test]
                        test_data = df['spanduk'][-n_test:]
                        model = ARIMA(train_data, order=(p, d, q))
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=n_test)

                        if show_accuracy:
                            mae = mean_absolute_error(test_data, forecast)
                            rmse = np.sqrt(mean_squared_error(test_data, forecast))
                            mape = mean_absolute_percentage_error(test_data, forecast) * 100

                            st.subheader("📏 Akurasi Model (berdasarkan 6 bulan terakhir)")
                            st.markdown(f"""
                            - **MAE**: {mae:.2f}
                            - **RMSE**: {rmse:.2f}
                            - **MAPE**: {mape:.2f}%
                            """)

                    else:
                        st.error("❌ Jumlah data terlalu sedikit untuk evaluasi (minimal 6 bulan data).")
                        forecast = pd.Series()

                    # Fit ulang ke seluruh data untuk prediksi 6 bulan ke depan
                    final_model = ARIMA(df['spanduk'], order=(p, d, q)).fit()
                    pred_next = final_model.forecast(steps=6)

                    st.subheader("📅 Hasil Prediksi 6 Bulan Kedepan")
                    st.write(pred_next)

                    fig, ax = plt.subplots()
                    df['spanduk'].plot(ax=ax, label='Data Historis')
                    pred_next.plot(ax=ax, label='Prediksi', color='orange')
                    ax.legend()
                    st.pyplot(fig)
        else:
            st.error("❌ Pastikan file CSV memiliki kolom: 'bulan' dan 'spanduk'")
