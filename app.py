# app.py - predicts mental health treatment-seeking likelihood in tech workers
# how to run:
#   1. make sure you have run main.py first so that .joblib files are created
#   2. install streamlit if needed:   pip install streamlit
#   3. from your terminal, run:       streamlit run app.py
#   4. a browser tab will open automatically at http://localhost:8501
#
# to deploy free online:
#   1. push your project folder to a github repo
#   2. go to https://share.streamlit.io
#   3. connect your github repo — it will auto-deploy in about 60 seconds

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import csv, datetime
import os

# page setup

st.set_page_config(
    page_title="Ruh Sağlığı Risk Tahmincisi",
    page_icon="🧠",
    layout="centered",
)

st.markdown(
    "<h1 style='text-align: center;'>🧠 Teknolojide Ruh Sağlığı</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center;'>Tedavi Arama Riski Tahmincisi</h3>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Aşağıdaki soruları mevcut iş durumunuza göre yanıtlayın. "
    "Model, profilinize sahip birinin ruh sağlığı tedavisi arama "
    "olasılığını tahmin edecektir.</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-style: italic;'>Yanıtlarınız anonim olarak kaydedilmektedir.</p>",
    unsafe_allow_html=True
)
st.divider()

# load saved model and feature columns

@st.cache_resource   # loads once, then reuses — keeps the app fast
def load_model():
    model   = joblib.load("Models/best_model.joblib")
    columns = joblib.load("Models/feature_columns.joblib")
    return model, columns

model, feature_columns = load_model()


# user input form

st.markdown("### Hakkınızda")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Yaşınız", min_value=15, max_value=75, value=28)

with col2:
    gender = st.selectbox("Cinsiyet", ["Erkek", "Kadın", "Diğer"])

# map turkish gender labels back to the values the model was trained on
gender_map = {"Erkek": "Male", "Kadın": "Female", "Diğer": "Other"}

self_employed = st.selectbox(
    "Serbest çalışıyor musunuz?", ["Hayır", "Evet"]
)

st.divider()
st.markdown("### İş yeriniz")

no_employees = st.selectbox(
    "Şirket büyüklüğü",
    ["1-5", "6-25", "26-100", "100-500", "500-1000", "1000'den fazla"]
)
# map the one label that differs from the original training value
no_employees_map = {"1000'den fazla": "More than 1000"}

tech_company = st.selectbox(
    "İşvereniniz öncelikli olarak bir teknoloji şirketi mi?", ["Evet", "Hayır"]
)

remote_work = st.selectbox(
    "Zamanınızın en az %50'sinde uzaktan çalışıyor musunuz?", ["Evet", "Hayır"]
)

st.divider()
st.markdown("### İş yerinde ruh sağlığı desteği")

benefits = st.selectbox(
    "İşvereniniz ruh sağlığı yardımları sunuyor mu?",
    ["Evet", "Hayır", "Bilmiyorum"]
)
yn_map = {"Evet": "Yes", "Hayır": "No", "Bilmiyorum": "Don't know"}

care_options = st.selectbox(
    "İşvereninizin sunduğu ruh sağlığı bakım seçeneklerini biliyor musunuz?",
    ["Evet", "Hayır", "Emin değilim"]
)
care_map = {"Evet": "Yes", "Hayır": "No", "Emin değilim": "Not sure"}

wellness_program = st.selectbox(
    "İşvereniniz ruh sağlığını resmi olarak ele aldı mı?",
    ["Evet", "Hayır", "Bilmiyorum"]
)

seek_help = st.selectbox(
    "İşvereniniz ruh sağlığı hakkında bilgi edinmek için kaynaklar sunuyor mu?",
    ["Evet", "Hayır", "Bilmiyorum"]
)

anonymity = st.selectbox(
    "Ruh sağlığı kaynaklarını kullanırsanız kimliğiniz korunuyor mu?",
    ["Evet", "Hayır", "Bilmiyorum"]
)

leave = st.selectbox(
    "Ruhsal sağlık nedeniyle hastalık izni almak ne kadar kolay?",
    ["Çok kolay", "Oldukça kolay", "Oldukça zor", "Çok zor", "Bilmiyorum"]
)
leave_map = {
    "Çok kolay":     "Very easy",
    "Oldukça kolay": "Somewhat easy",
    "Oldukça zor":   "Somewhat difficult",
    "Çok zor":       "Very difficult",
    "Bilmiyorum":    "Don't know",
}

st.divider()
st.markdown("### İş yeri kültürü")

mental_health_consequence = st.selectbox(
    "İşvereninizle ruh sağlığı sorununu konuşmak olumsuz sonuçlar doğurur mu?",
    ["Evet", "Hayır", "Belki"]
)
consequence_map = {"Evet": "Yes", "Hayır": "No", "Belki": "Maybe"}

coworkers = st.selectbox(
    "Bir ruh sağlığı sorununu iş arkadaşlarınızla paylaşır mıydınız?",
    ["Evet", "Hayır", "Bazılarıyla"]
)
some_map = {"Evet": "Yes", "Hayır": "No", "Bazılarıyla": "Some of them"}

supervisor = st.selectbox(
    "Bir ruh sağlığı sorununu amirinizle paylaşır mıydınız?",
    ["Evet", "Hayır", "Bazılarıyla"]
)

obs_consequence = st.selectbox(
    "Ruh sağlığı sorunu olan iş arkadaşlarının olumsuz sonuçlarla karşılaştığını gördünüz mü?",
    ["Evet", "Hayır"]
)
obs_map = {"Evet": "Yes", "Hayır": "No"}

st.divider()
st.markdown("### Kişisel geçmiş")

family_history = st.selectbox(
    "Ailenizde ruhsal hastalık öyküsü var mı?",
    ["Evet", "Hayır"]
)

work_interfere = st.selectbox(
    "Bir ruh sağlığı durumunuz varsa, bu işinizi ne sıklıkla etkiliyor?",
    ["Hiçbir zaman", "Nadiren", "Bazen", "Sıkça", "Bilinmiyor"]
)
interfere_map = {
    "Hiçbir zaman": "Never",
    "Nadiren":      "Rarely",
    "Bazen":        "Sometimes",
    "Sıkça":        "Often",
    "Bilinmiyor":   "Unknown",
}


# build input row and predict

def build_input_row():
    """turn the user's dropdown choices into the same format the model was trained on."""
    data = {
        "Age":                        age,
        "Gender":                     gender_map.get(gender, gender),
        "self_employed":              1 if self_employed == "Evet" else 0,
        "no_employees":               no_employees_map.get(no_employees, no_employees),
        "tech_company":               1 if tech_company == "Evet" else 0,
        "remote_work":                1 if remote_work == "Evet" else 0,
        "benefits":                   yn_map.get(benefits, benefits),
        "care_options":               care_map.get(care_options, care_options),
        "wellness_program":           yn_map.get(wellness_program, wellness_program),
        "seek_help":                  yn_map.get(seek_help, seek_help),
        "anonymity":                  yn_map.get(anonymity, anonymity),
        "leave":                      leave_map.get(leave, leave),
        "mental_health_consequence":  consequence_map.get(mental_health_consequence, mental_health_consequence),
        "coworkers":                  some_map.get(coworkers, coworkers),
        "supervisor":                 some_map.get(supervisor, supervisor),
        "obs_consequence":            obs_map.get(obs_consequence, obs_consequence),
        "family_history":             obs_map.get(family_history, family_history),
        "work_interfere":             interfere_map.get(work_interfere, work_interfere),
    }

    row = pd.DataFrame([data])

    # one-hot encode to match training format
    row = pd.get_dummies(row)

    # add any missing columns (with value 0) and drop any extra ones
    for col in feature_columns:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_columns]

    return row


# predict button

st.divider()

if st.button("🔍  Risk skorumu tahmin et", use_container_width=True):
    input_row    = build_input_row()
    probability  = model.predict_proba(input_row)[0][1]   # probability of seeking treatment
    risk_percent = round(probability * 100)

    st.markdown("## Sonucunuz")

    # colour-coded risk band
    if risk_percent >= 70:
        colour = "#D85A30"
        label  = "Yüksek olasılık"
        message = (
            "Binlerce teknoloji çalışanından elde edilen anket verilerine göre "
            "profilinize sahip birinin ruh sağlığı tedavisi arama olasılığı yüksektir. "
            "Bu bir tanı değildir — iş yeri ve kişisel risk faktörlerini yansıtmaktadır."
        )
    elif risk_percent >= 40:
        colour = "#BA7517"
        label  = "Orta olasılık"
        message = (
            "Profilinize sahip birinin tedavi arama olasılığı orta düzeydedir. "
            "İş yerinizde bazı koruyucu faktörler mevcut olsa da belirli risk "
            "faktörleri devam etmektedir."
        )
    else:
        colour = "#0F6E56"
        label  = "Düşük olasılık"
        message = (
            "Mevcut iş yeri koşullarına göre profilinize sahip birinin tedavi arama "
            "olasılığı daha düşüktür. Güçlü işveren desteği koruyucu bir etken "
            "olarak görünmektedir."
        )

    st.markdown(
        f"""
        <div style="
            background: {colour}18;
            border-left: 4px solid {colour};
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        ">
            <p style="font-size: 2.5rem; font-weight: 600; margin: 0; color: {colour};">
                {risk_percent}%
            </p>
            <p style="font-size: 1rem; font-weight: 500; margin: 4px 0 0; color: {colour};">
                {label}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(message)

    st.caption(
        "Model: OSMI Teknoloji Sektöründe Ruh Sağlığı Anketi (2014 + 2016) üzerinde "
        "ayarlanmış Rastgele Orman. Bu araç yalnızca eğitim amaçlıdır."
    )

    # show the top 3 features driving the prediction
    st.divider()
    st.markdown("### Bu tahmini en çok etkileyen faktörler")
    importances   = model.feature_importances_
    top3_idx      = np.argsort(importances)[::-1][:3]
    top3_features = [(feature_columns[i], round(importances[i] * 100, 1)) for i in top3_idx]

    for feat, imp in top3_features:
        readable = feat.replace("_", " ").replace(" Yes", " → Evet").replace(" No", " → Hayır")
        st.markdown(f"- **{readable}** — önem skoru {imp}%")

    st.info(message)
    st.caption("Tüm yanıtlar kaydedilmiştir.")

    # save response to csv, raw inputs, no country and gender as binary columns
    response = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "age": age,
        "male": 1 if gender == "Erkek" else 0,
        "female": 1 if gender == "Kadın" else 0,
        "self_employed": 1 if self_employed == "Evet" else 0,
        "family_history": 1 if family_history == "Evet" else 0,
        "work_interfere": interfere_map.get(work_interfere),
        "no_employees": no_employees_map.get(no_employees, no_employees).replace("-", "_"),
        "remote_work": 1 if remote_work == "Evet" else 0,
        "tech_company": 1 if tech_company == "Evet" else 0,
        "benefits": yn_map.get(benefits),
        "care_options": care_map.get(care_options),
        "wellness_program": yn_map.get(wellness_program),
        "seek_help": yn_map.get(seek_help),
        "anonymity": yn_map.get(anonymity),
        "leave": leave_map.get(leave),
        "mental_health_consequence": consequence_map.get(mental_health_consequence),
        "coworkers": some_map.get(coworkers),
        "supervisor": some_map.get(supervisor),
        "obs_consequence": obs_map.get(obs_consequence),
        "predicted_probability": round(probability, 4),
    }

    save_path = "Saved Respones/responses.csv"
    response_df = pd.DataFrame([response])
    response_df.to_csv(save_path, mode="a", header=not os.path.exists(save_path), index=False)

