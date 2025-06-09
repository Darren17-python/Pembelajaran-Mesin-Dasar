import numpy as np
import pandas as pd
import joblib
import pickle
import streamlit as st

def hitung_jarak_manhattan(x1, x2):
    return np.sum(np.abs(x1 - x2))

def knn(X_train, y_train, X_test, k, distance_func):
    distances = []
    for i in range(len(X_train)):
        dist = distance_func(X_train[i], X_test[0])
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest = [label for (_, label) in distances[:k]]
    prediction = max(set(k_nearest), key=k_nearest.count)
    return [prediction]

X_train_full = pd.read_csv("X_train.csv").values
y_train = pd.read_csv("y_train.csv").values.ravel()
scaler = joblib.load("scaler.pkl")

with open("model_knn_aco.pkl", "rb") as file:
    model = pickle.load(file)

selected_indices = model["selected_indices"]
k = model["k"]

label_map = {0: "Low", 1: "Medium", 2: "High"}

X_train = X_train_full[:, selected_indices]

st.title("Prediksi Popularitas Konten Media Sosial (KNN)")

platform_input = st.selectbox("Pilih Platform:", ["TikTok", "Instagram", "YouTube", "Twitter"])
platform_mapping = {'TikTok': 0, 'Instagram': 1, 'YouTube': 2, 'Twitter': 3}
platform = platform_mapping[platform_input]

views = st.number_input("Views", min_value=0, value=1500, step=100, format="%d")
likes = st.number_input("Likes", min_value=0, value=500, step=100, format="%d")
shares = st.number_input("Shares", min_value=0, value=100, step=10, format="%d")
comments = st.number_input("Comments", min_value=0, value=50, step=10, format="%d")

hashtag_options = ["Fashion", "Education", "Comedy", "Tech", "Viral", "Challenge", "Fitness", "Music", "Gaming", "Dance"]
hashtag_group_input = st.selectbox("Pilih Hashtag Group:", hashtag_options)

region_options = ["UK", "India", "Brazil", "Australia", "Japan", "Germany", "Canada", "USA"]
region_group_input = st.selectbox("Pilih Region Group:", region_options)

content_options = ["Tweet", "Reel", "Live Stream", "Video", "Post", "Short"]
content_group_input = st.selectbox("Pilih Content Group:", content_options)

hashtag_group = hashtag_options.index(hashtag_group_input) / (len(hashtag_options) - 1)
region_group = region_options.index(region_group_input) / (len(region_options) - 1)
content_group = content_options.index(content_group_input) / (len(content_options) - 1)

if st.button("Prediksi"):
    input_numerik = np.array([[views, likes, shares, comments, hashtag_group, region_group, content_group]])
    input_scaled = scaler.transform(input_numerik)

    input_final = np.concatenate(([platform], input_scaled[0])).reshape(1, -1)
    input_selected = input_final[:, selected_indices]
    prediction = knn(X_train, y_train, input_selected, k, hitung_jarak_manhattan)
    prediksi_kelas = label_map[prediction[0]]

    st.success(f"Prediksi kelas popularitas konten: **{prediksi_kelas}**")

st.markdown("### Keterangan Kelas Popularitas")
st.markdown("- **Low**: Konten dengan popularitas rendah")
st.markdown("- **Medium**: Konten dengan popularitas sedang")
st.markdown("- **High**: Konten dengan popularitas tinggi")
