import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Porter Stemmer
ps = PorterStemmer()

# Define text transformation function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text
    y = [i for i in text if i.isalnum()]  # Remove non-alphanumeric characters

    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]  # Remove stopwords and punctuation

    y = [ps.stem(i) for i in y]  # Stem the words

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set page configuration
st.set_page_config(page_title="Spam Classifier", page_icon="✉️", layout="centered", initial_sidebar_state="auto")

# Add custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .title {
            font-size: 2.5rem;
            color: #4B7BEC;
            text-align: center;
        }
        .subtitle {
            font-size: 1.25rem;
            color: #333333;
            text-align: center;
            margin-top: -1rem;
            margin-bottom: 2rem;
        }
        .input-area {
            font-size: 1rem;
            color: #333333;
            margin-bottom: 1rem;
            padding: 0.75rem;
            border: 2px solid #4B7BEC;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .predict-button {
            background-color: #4B7BEC;
            color: white;
            font-size: 1.25rem;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .predict-button:hover {
            background-color: #3a5bbf;
        }
        .result {
            font-size: 1.5rem;
            margin-top: 1rem;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Main content container
with st.container():
    st.markdown("<h1 class='title'>E-mail/SMS Spam Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Classify your messages with ease</p>", unsafe_allow_html=True)

    # Input text area
    input_sms = st.text_area("Enter your message here:", height=150, help="Type or paste your message to check if it's spam or not.", placeholder="Type your message...")

    # Predict button
    if st.button('Predict', key='predict-button', help="Click to predict if the message is spam or not."):
        # Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # Vectorize the transformed text
        vector_input = tfidf.transform([transformed_sms])

        # Predict the result
        result = model.predict(vector_input)[0]

        # Display the result
        if result == 1:
            st.markdown("<h2 class='result' style='color: red;'>Prediction: 🗑️ The above entered message is predicted to be a spam</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 class='result' style='color: green;'>Prediction: 📧 The above entered message is predicted to be a ham</h2>", unsafe_allow_html=True)
