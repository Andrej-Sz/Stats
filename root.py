import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import helpers as hl

# Check if DataFrame exists in session state
df = st.session_state.get("df")

if df is not None:
    with st.expander("CSV Head"):
        st.dataframe(df.head())

    with st.expander("Describe"):
        st.dataframe(df.describe())

    with st.expander("Correlation"):
        corr_cols = st.multiselect("Columns", df.columns, key="corr_cols")
        corr_figsize = st.slider("FigSize", min_value=5, max_value=15, value=6)

        if corr_cols:
            corr_plt, corr_ax = plt.subplots(figsize=(corr_figsize+2, corr_figsize))
            sns.heatmap(df[corr_cols].corr().round(2), annot=True, cmap="RdBu_r", vmin=-1, vmax=1, ax=corr_ax)
            st.pyplot(corr_plt.figure)
            plt.close()

    with st.expander("Box Plot"):
        bx_cols = st.multiselect("Columns", df.columns, key="bx_cols")

        bx_c1, bx_c2, bx_c3 = st.columns([1,3,3])
        with bx_c1:
            bx_orientation = st.selectbox("Orientation", options=["H", "V"])
        with bx_c2:
            bx_figsize_x = st.slider("FigSize X", min_value=1, max_value=10, value=8)
        with bx_c3:
            bx_figsize_y = st.slider("FigSize Y", min_value=1, max_value=10, value=4)

        if bx_cols:
            bx_plt, bx_ax = plt.subplots(figsize=(bx_figsize_x, bx_figsize_y))
            sns.boxplot(
                df[bx_cols], showfliers=True, showmeans=True, ax=bx_ax, 
                orient="h" if bx_orientation == "H" else "v",
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#111827", "markersize": 6},
                flierprops={"marker": "o", "markersize": 3.5, "markerfacecolor": "#6b7280", "markeredgecolor": "none", "alpha": 0.7}
            )
            st.pyplot(bx_plt.figure)
            plt.close()

    with st.expander("Violin Plot"):
        vl_cols = st.multiselect("Columns", df.columns, key="vl_cols")

        vl_c1, vl_c2 = st.columns([1,1])
        with vl_c1:
            vl_figsize_x = st.slider("FigSize X", min_value=1, max_value=10, value=8, key="vl_x")
        with vl_c2:
            vl_figsize_y = st.slider("FigSize Y", min_value=1, max_value=10, value=4, key="vl_y")

        if vl_cols:
            vl_plot, vl_ax = plt.subplots(figsize=(vl_figsize_x, vl_figsize_y))
            sns.violinplot(df[vl_cols], ax=vl_ax)
            st.pyplot(vl_plot.figure)
            plt.close()

    with st.expander("Matrix Plot"):
        pair_cols = st.multiselect("Columns", df.columns, key="pair_cols")

        pair_c1, pair_c2, pair_c3 = st.columns([1,1,5])
        with pair_c1:
            pair_kind = st.selectbox("Matrix Kind", options=["scatter", "kde", "hist", "reg"])
        with pair_c2:
            pair_diag_kind = st.selectbox("Diagonal Kind", options=["auto", "hist", "kde"])
        with pair_c3:
            pair_figsize = st.slider("FigSize", min_value=1, max_value=10, value=5)

        if pair_cols:
            pair_plt = sns.pairplot(df[pair_cols], kind=pair_kind, diag_kind=pair_diag_kind, size=pair_figsize)
            st.pyplot(pair_plt.figure)
            plt.close()
else:
    st.warning("Please upload a CSV file in the sidebar to proceed.")