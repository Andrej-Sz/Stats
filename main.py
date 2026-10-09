import chardet
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

with st.sidebar:
    uploaded = st.file_uploader("Upload a CSV", type=["csv"], key="csv_uploader")
    if uploaded is not None:
        raw_data = uploaded.read(10000)
        detected = chardet.detect(raw_data)
        encoding = detected.get("encoding") or "utf-8"
        uploaded.seek(0)

        try:
            st.session_state["df"] = pd.read_csv(uploaded, encoding=encoding)
        except UnicodeDecodeError:
            uploaded.seek(0)
            st.session_state["df"] = pd.read_csv(uploaded, encoding="latin1")
    else:
        if "df" not in st.session_state:
            st.info("Choose a CSV file to preview the first rows.")

main_page = st.Page("root.py", title="Exploratory Analysis", default=True)
pca_page = st.Page("PCA.py", title="PCA Analysis")

pg = st.navigation([main_page, pca_page])
pg.run()
