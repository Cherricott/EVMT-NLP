import ctranslate2
import streamlit as st
import sentencepiece as spm
import nltk
from nltk import sent_tokenize
import torch

torch.classes.__path__ = []
nltk.download("punkt_tab", quiet=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ct_model_path = "evb-opennmt/model"
sp_source_model_path = "evb-opennmt/model/source.model"
sp_target_model_path = "evb-opennmt/model/target.model"

translator = ctranslate2.Translator(ct_model_path, DEVICE)
sp_source_model = spm.SentencePieceProcessor(sp_source_model_path)
sp_target_model = spm.SentencePieceProcessor(sp_target_model_path)


st.set_page_config(page_title="OpenNMT", page_icon="📖")
st.title("OpenNMT Translate")

with st.form("my_form"):
    # Text area for source text
    user_input = st.text_area(
        "Source Text", max_chars=200, placeholder="Example: I have a cat!"
    )
    source_sentences = sent_tokenize(user_input)
    # Encode source tokens
    source_tokenized = sp_source_model.encode(source_sentences, out_type=str)
    # Translate
    translations = translator.translate_batch(source_tokenized)
    translations = [translation.hypotheses[0] for translation in translations]
    # Decode target tokens
    translations_detokenized = sp_target_model.decode(translations)
    translation = " ".join(translations_detokenized)

    submitted = st.form_submit_button("Translate")
    if submitted:
        st.write("Translation")
        st.info(translation)
