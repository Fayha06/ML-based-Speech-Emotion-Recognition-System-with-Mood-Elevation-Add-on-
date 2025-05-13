import streamlit as st
import numpy as np
import librosa
from joblib import load
import pandas as pd

# Load the trained model
model_path = 'Random Forest_model.joblib'  # Change if needed
model = load(model_path)

# Emotion labels used in training
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


# Feature extraction function (same as your training code)
def extract_feature(file, mfcc=True, chroma=True, mel=True):
    try:
        X, sample_rate = librosa.load(file, res_type='kaiser_fast')
        result = np.array([])

        if chroma:
            stft = np.abs(librosa.stft(X))

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))

        return result
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None


# Mood Enhancement Suggestions based on predicted emotion
def get_mood_enhancement_suggestions(emotion):
    try:
        mood_enhancement_df = pd.read_csv('mood_enhance.csv')

        # Filter the dataset based on the predicted emotion
        filtered_df = mood_enhancement_df[mood_enhancement_df['Mood'].str.lower() == emotion.lower()]

        if filtered_df.empty:
            return pd.DataFrame(columns=['Category', 'Name', 'YouTube_Link'])

        return filtered_df[['Category', 'Name', 'YouTube_Link']]
    except FileNotFoundError:
        st.warning("mood_enhancement.csv not found. Skipping mood enhancement suggestions.")
        return pd.DataFrame(columns=['Category', 'Name', 'YouTube_Link'])

# Streamlit UI
st.set_page_config(page_title="🎙️ Speech Emotion Recognition", layout="centered")
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a WAV audio file to predict the speaker's emotion.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("Analyzing audio and predicting emotion..."):
        features = extract_feature(uploaded_file)

        if features is None or features.shape[0] != 180:
            st.error("Feature extraction failed or invalid audio file.")
        else:
            prediction = model.predict([features])[0]
            st.success(f"Predicted Emotion: **{prediction.capitalize()}**")

            # Display mood-enhancement suggestions
            st.subheader("🌟 Mood Enhancement Suggestions")
            suggestions = get_mood_enhancement_suggestions(prediction)

            if not suggestions.empty:
                emoji_map = {
                    'Listen': '🎵 Listen',
                    'Activity': '🏃 Activity',
                    'Self-Improve': '📚 Inspire'
                }

                st.info("Here are suggestions based on the detected emotion to help uplift your mood.")

                category_order = ['Listen', 'Activity', 'Self-Improve']

                for category in category_order:
                    category_df = suggestions[suggestions['Category'] == category]

                    if not category_df.empty:
                        st.markdown(f"### {emoji_map.get(category, category)}")

                        for _, row in category_df.iterrows():
                            st.markdown(f"- **{row['Name']}** - [YouTube Link]({row['YouTube_Link']})")


            else:
                st.info("No mood enhancement suggestions available for this emotion.")




