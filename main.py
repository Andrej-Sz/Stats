import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# Render File Uploader in Sidebar on ALL pages
with st.sidebar:
    uploaded = st.file_uploader("Upload a CSV", type=["csv"], key="csv_uploader")
    if uploaded is not None:
        # Cache the DataFrame into Session State
        st.session_state["df"] = pd.read_csv(uploaded)
    else:
        if "df" not in st.session_state:
            st.info("Choose a CSV file to preview the first rows.")

# Configure Multi-Page Navigation
main_page = st.Page("root.py", title="Exploratory Analysis", icon="📊", default=True)
pca_page = st.Page("PCA.py", title="PCA Analysis", icon="📈")

pg = st.navigation([main_page, pca_page])
pg.run()