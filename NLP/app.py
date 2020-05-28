import spacy
import streamlit as st
import pandas as pd
import joblib, os
from spacy.lang.en import English
nlp = spacy.load("en_core_web_sm")

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Vectorizer
new_vector = open("models/final_news_cv_vectorizer.pkl", "rb")
new_cv = joblib.load(new_vector)


# Load Model
def load_prediction_model(model_file):
    load_model = joblib.load(open(os.path.join(model_file), "rb"))
    return load_model


def get_Keys(vals, my_dict):
    for key, value in my_dict.items():
        if vals == value:
            return key


def main():
    "Streamlit"
    st.title("News Classifier")
    activity = ["Prediction", "NLP"]
    choice = st.sidebar.selectbox("Choose Activity", activity)

    if choice == "Prediction":
        st.info("Prediction with ML")

        news_text = st.text_area("Enter Text", "Type Here")
        ml_models = ["LR", "NB", "RForest", "Decision Tree"]
        model_choice = st.selectbox("Choose ML Model", ml_models)
        prediction_labels = {"Business": 0, "tech": 1, "sport": 2, "health": 3, "politics": 4,
                             "entertaiment": 5}
        if st.button("Classify"):
            st.text("Original Text :: \n{}".format(news_text))
            vector_text = new_cv.transform([news_text]).toarray()
            if model_choice == "LR":
                predictor = load_prediction_model("models/newsclassifier_Logit_model.pkl")
                precdiction = predictor.predict(vector_text)
            # st.write(precdiction)

            if model_choice == "NB":
                predictor = load_prediction_model("models/newsclassifier_NB_model.pkl")
                precdiction = predictor.predict(vector_text)
            # st.write(precdiction)

            if model_choice == "RForest":
                predictor = load_prediction_model("models/newsclassifier_RFOREST_model.pkl")
                precdiction = predictor.predict(vector_text)
            # st.write(precdiction)

            if model_choice == "Decision Tree":
                predictor = load_prediction_model("models/newsclassifier_CART_model.pkl")
                precdiction = predictor.predict(vector_text)
                # st.write(precdiction)

            final_result = get_Keys(precdiction, prediction_labels)
            st.success("News Category as : {}".format(final_result))

    if choice == "NLP":
        st.info("Natural Language Processing")
        news_text = st.text_area("Enter Text", "Type Here")
        nlp_task = ["Tokenization", "NER", "Lemmatization"]
        task_choice = st.selectbox("Choose NLP Task", nlp_task)
        if st.button("Analyze"):
            st.info("Original Text".format(news_text))

            docx = nlp(news_text)
            if task_choice == "Tokenization":
                result = [token.text for token in docx]
            elif task_choice == "NER":
                result = [(entity.text, entity.label_) for entity in docx.ents]
            elif task_choice == "Lemmatization":
                result = ["'Token' : {},'Lemma':{}".format(token.text, token.lemma_) for token in docx]
            elif task_choice == "POS Tags":
                result = ["'Token ':{}, 'Pos' : {}, 'Dependency' : {}".format(word.text, word.tag_, word.dep_) for word
                          in docx]
            st.json(result)

        if st.button("Tabulize"):
            docx = nlp(news_text)
            c_token = [token.text for token in docx]
            c_lemma = [(token.lemma_) for token in docx]
            c_pos = [(word.tag_, word.dep_) for word in docx]

            new_df = pd.DataFrame(zip(c_token, c_lemma, c_pos), columns=["Token", "Lemma", "POS"])
            st.dataframe(new_df)

        # Wordcloud
        if st.checkbox("Wordcloud"):
            wordcloud = WordCloud().generate(news_text)
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot()


if __name__ == '__main__':
    main()
