import os
import joblib
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from docx import Document


# Fungsi untuk memuat dan membaca file Word
def read_word_file(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Path untuk memuat dan menyimpan model
model_paths = {
    "Random Forest": "random_forest.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "K-Nearest Neighbors": "k-nearest_neighbors.pkl"
}

# Fungsi untuk menyimpan model
def save_model(model, model_name):
    joblib.dump(model, model_paths[model_name])

# Fungsi untuk memuat model
def load_model(model_name):
    if os.path.exists(model_paths[model_name]):
        return joblib.load(model_paths[model_name])
    else:
        st.warning(f"Model {model_name} tidak ditemukan. Latih dan simpan model terlebih dahulu.")
        return None

def main():

    st.title("Diabetes Prediction Using Multiple Algorithms")

    file_path = "Dataset of Diabetes .csv" #path ke dataset
    word_file_path = "penjelasan dataset.docx"

    try:
        # Membaca dataset
        df = pd.read_csv(file_path)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        if os.path.exists(word_file_path):
                # Menampilkan tombol download untuk file Word
                with open(word_file_path, "rb") as file:
                    st.download_button(
                        label="Download Dataset Explanation",
                        data=file,
                        file_name="Dataset_Explanation.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
        else:
                st.warning("File penjelasan dataset tidak ditemukan.")
        
        st.header("Model Training and Prediction")
        
        # Input manual untuk prediksi
        st.subheader("Predict Diabetes Class")

        # Pilih model untuk prediksi
        model_choice = st.selectbox("Choose Model", options=["Random Forest", "Logistic Regression", "K-Nearest Neighbors"])

        # Memuat model yang dipilih
        model = load_model(model_choice)

        if model:
            gender = st.selectbox("Gender (M/F)", options=["Male", "Female"])
            age = st.number_input("AGE (tahun)")
            urea = st.number_input("Urea (mg/dL)")
            cr = st.number_input("Cr (Creatinine ratio) ")
            hba1c = st.number_input("HbA1c Level (%)")
            chol = st.number_input("Chol (Cholesterol)  (mg/dL)")
            tg = st.number_input("TG (Triglycerides) (mg/dL)")
            hdl = st.number_input("HDL Cholesterol (mg/dL)")
            ldl = st.number_input("LDL Cholesterol (mg/dL)")
            vldl = st.number_input("VLDL Cholesterol (mg/dL)")
            bmi = st.number_input("BMI")

            if st.button("Predict"):
                # Konversi input ke format model
                input_data = pd.DataFrame({
                    "Gender": [1 if gender == "Male" else 0],
                    "AGE": [age],
                    "Urea": [urea],
                    "Cr": [cr],
                    "HbA1c": [hba1c],
                    "Chol": [chol],
                    "TG": [tg],
                    "HDL": [hdl],
                    "LDL": [ldl],
                    "VLDL": [vldl],
                    "BMI": [bmi],
                })

                # Prediksi dengan model yang dipilih
                prediction = model.predict(input_data)[0]
                
                if prediction == "N":
                    st.write("Predicted Class: **Tidak Menderita Diabetes**")
                elif prediction == "P":
                    st.write("Predicted Class: **Kemungkinan Besar Menderita Diabetes**")
                elif prediction == "Y":
                    st.write("Predicted Class: **Menderita Diabetes**")
                else:
                    st.write("Predicted Class: Tidak Diketahui")


            
    except FileNotFoundError:
        st.error(f"Dataset file not found at path: {file_path}. Please check the path and try again.")

if __name__ == "__main__":
    main()
