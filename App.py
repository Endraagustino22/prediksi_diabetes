import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Judul aplikasi
st.title("Random Forest Classifier for Diabetes Prediction")

# Fungsi utama aplikasi
def main():
    st.header("Dataset Import and Model Training")

    # Path ke dataset
    file_path = "Dataset of Diabetes .csv"  # Ganti dengan path dataset Anda

    try:
        # Membaca dataset
        df = pd.read_csv(file_path)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        # Data preprocessing
        df['CLASS'] = df['CLASS'].str.strip()
        df['Gender'] = df['Gender'].str.upper()
        df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
        df = df.rename(columns={
            "Age": "AGE",
            "Cholesterol": "Chol",
            "Creatinine ratio (Cr)": "Cr",
            "HBA1C": "HbA1c"
        })
        df = df.drop(['ID', 'No_Pation'], axis=1)

        # Memisahkan fitur dan label
        X = df.drop(['CLASS'], axis=1)  # fitur
        y = df['CLASS']  # label

        # Pembagian data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Melatih model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluasi model
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        st.subheader("Model Performance")
        st.write(f"Training Accuracy: {train_accuracy:.2f}")
        st.write(f"Testing Accuracy: {test_accuracy:.2f}")

        # Menampilkan laporan klasifikasi
        st.subheader("Classification Report")
        report = classification_report(y_test, y_test_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Input manual untuk prediksi
        st.subheader("Predict Diabetes Class")
        gender = st.selectbox("Gender", options=["Male", "Female"])
        age = st.number_input("AGE", max_value=120, step=1)
        urea = st.number_input("Urea", max_value=150, step=1)
        cr = st.number_input("Cr (Creatinine ratio)", step=0.1)
        hba1c = st.number_input("HbA1c Level (%)", max_value=15.0, step=0.1)
        chol = st.number_input("Chol (Cholesterol)", max_value=400, step=1)
        tg = st.number_input("TG (Triglycerides)", max_value=400, step=1)
        hdl = st.number_input("HDL Cholesterol", max_value=100, step=1)
        ldl = st.number_input("LDL Cholesterol", max_value=250, step=1)
        vldl = st.number_input("VLDL Cholesterol", max_value=100, step=1)
        bmi = st.number_input("BMI",max_value=60.0, step=0.1)

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

            # Prediksi
            prediction = model.predict(input_data)[0]
            st.write(f"Predicted Class: **{prediction}**")

    except FileNotFoundError:
        st.error(f"Dataset file not found at path: {file_path}. Please check the path and try again.")

if __name__ == "__main__":
    main()
