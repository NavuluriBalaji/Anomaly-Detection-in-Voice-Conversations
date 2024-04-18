import streamlit as st
import pandas as pd
import numpy as np
import librosa
from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.cof import COF

def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path)
    except:
        st.error("Error: Failed to load audio file.")
        return None
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    energy = librosa.feature.rms(y=y)[0]
    
    features = np.vstack([mfccs, spectral_centroid, spectral_bandwidth, zero_crossing_rate, energy])
    return features.T

def detect_anomaly_pca(features):
    pca = PCA(n_components=10)
    pca.fit(features)
    features_compressed = pca.transform(features)
    distance = np.linalg.norm(features_compressed)
    threshold = np.std(features_compressed) * 2
    return distance > threshold

def detect_anomaly_zscore(features):
    z_scores = zscore(features)
    return np.any(np.abs(z_scores) > 3)

def detect_anomaly_isolation_forest(features):
    clf = IsolationForest(contamination=0.05)
    y_pred = clf.fit_predict(features)
    return -1 in y_pred

def detect_anomaly_lof(features):
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    y_pred = clf.fit_predict(features)
    return -1 in y_pred

# Streamlit App
st.title("Anomaly Detection in Voice Conversations")

st.sidebar.title("Extracted Features")
st.sidebar.write("No features extracted yet.")

uploaded_file = st.sidebar.file_uploader("Upload Audio File (WAV format)")

if uploaded_file is not None:
    features = extract_audio_features(uploaded_file)
    if features is not None:
        st.sidebar.write("### Extracted Features")
        df_features = pd.DataFrame(features, columns=["Feature {}".format(i+1) for i in range(features.shape[1])])
        st.sidebar.write(df_features)

# Main content
st.markdown("---")
st.header("Choose Anomaly Detection Method")

selected_method = st.selectbox("Select Method", ["PCA", "Z-score", "Isolation Forest", "Local Outlier Factor"])

if selected_method == "PCA":
    st.subheader("Principal Component Analysis (PCA)")
    st.write("PCA is a technique used for dimensionality reduction. It identifies the most important features in the data by projecting it onto a lower-dimensional subspace.")
    st.write("PCA can be used for anomaly detection by comparing the distance of data points from the reference distribution in the reduced feature space.")
elif selected_method == "Z-score":
    st.subheader("Z-score")
    st.write("The Z-score measures how many standard deviations a data point is from the mean of the dataset. It is commonly used to identify outliers in the data.")
    st.write("Anomaly detection using Z-score involves identifying data points that have a Z-score greater than a certain threshold.")
elif selected_method == "Isolation Forest":
    st.subheader("Isolation Forest")
    st.write("Isolation Forest is an ensemble learning algorithm that isolates anomalies by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.")
    st.write("Anomalies are expected to have shorter path lengths in the tree structure, making them easier to isolate.")
elif selected_method == "Local Outlier Factor":
    st.subheader("Local Outlier Factor (LOF)")
    st.write("LOF is a density-based algorithm that compares the density of data points in the vicinity of each point to the density of its neighbors.")
    st.write("Anomalies are identified as data points with significantly lower density compared to their neighbors.")

# Detect Anomaly button
if uploaded_file is not None and st.button("Detect Anomaly"):
    if features is None:
        st.error("Error: Features not extracted. Please upload a valid audio file.")
    else:
        if selected_method == "PCA":
            if detect_anomaly_pca(features):
                st.write("Anomaly detected!")
            else:
                st.write("No anomaly detected.")
        elif selected_method == "Z-score":
            if detect_anomaly_zscore(features):
                st.write("Anomaly detected!")
            else:
                st.write("No anomaly detected.")
        elif selected_method == "Isolation Forest":
            if detect_anomaly_isolation_forest(features):
                st.write("Anomaly detected!")
            else:
                st.write("No anomaly detected.")
        elif selected_method == "Local Outlier Factor":
            if detect_anomaly_lof(features):
                st.write("Anomaly detected!")
            else:
                st.write("No anomaly detected.")

# Description of advantages of detecting anomalies in voice conversations
st.markdown("---")
st.header("Advantages of Detecting Anomalies in Voice Conversations")
st.write("Detecting anomalies in voice conversations can provide several benefits, including:")
st.write("- Early detection of fraudulent activities, such as voice spoofing or impersonation.")
st.write("- Protection against data breaches and unauthorized access to sensitive information.")
st.write("- Improved quality assurance by identifying and resolving issues in voice communication systems.")
st.write("- Enhanced security and privacy for users, ensuring their personal data is protected.")
