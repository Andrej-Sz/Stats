import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import helpers as hl

# Fetch DataFrame from session state
df = st.session_state.get("df")

if df is not None:
    st.title("PCA Analysis")
    
    pca_cols = st.multiselect("Columns", df.columns, key="pca_cols")
    pca_n_comps_ = st.slider("N Principal Components", value=3, min_value=1, max_value=10)

    if pca_cols:
        pca_n_comps = min(pca_n_comps_, len(pca_cols))
        raw = PCA(n_components=pca_n_comps, random_state=0).fit(df[pca_cols])
        scaled = PCA(n_components=pca_n_comps, random_state=0).fit(StandardScaler().fit_transform(df[pca_cols]))

        st.dataframe(pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(pca_n_comps)],
            "unscaled share": raw.explained_variance_ratio_,
            "scaled share": scaled.explained_variance_ratio_,
        }).round(4))

        with st.expander("Individual Principal Components"):
            pca_idx = st.slider("Principal Component X Loading", min_value=1, max_value=pca_n_comps, value=1)
            pca_idx_ = pca_idx - 1
            st.dataframe(pd.DataFrame({
                "component": pca_cols,
                "unscaled PC": raw.components_[pca_idx_],
                "scaled PC": scaled.components_[pca_idx_],
            }).round(4))

            pca_comp_c1, pca_comp_c2 = st.columns([1,1])

            with pca_comp_c1:
                pca_comp_plt1, pca_comp_ax1 = plt.subplots(figsize=(5,5))
                pca_comp_ax1.barh(pca_cols, raw.components_[pca_idx_])
                pca_comp_ax1.set_title(f"PC{pca_idx} Unscaled")
                pca_comp_ax1.axvline(0, color='black', linewidth=1, linestyle='--')
                pca_comp_ax1.set_xlim(-1, 1)
                st.pyplot(pca_comp_plt1)
                plt.close()
            with pca_comp_c2:
                pca_comp_plt2, pca_comp_ax2 = plt.subplots(figsize=(5,5))
                pca_comp_ax2.barh(pca_cols, scaled.components_[pca_idx_])
                pca_comp_ax2.set_title(f"PC{pca_idx} Scaled")
                pca_comp_ax2.axvline(0, color="black", linewidth=1, linestyle="--")
                pca_comp_ax2.set_xlim(-1, 1)
                st.pyplot(pca_comp_plt2)
                plt.close()

        with st.expander("Principal Components Variance Share"):
            share_cum_df = pd.DataFrame({
                "component":[ f"PC{i+1}" for i in range(pca_n_comps)],
                "share": scaled.explained_variance_ratio_,
                "cumulative": np.cumsum(scaled.explained_variance_ratio_)
            })
            st.dataframe(share_cum_df)

            pca_pareto, pca_pareto_ax = plt.subplots(figsize=(8,4))
            pca_pareto_ax.bar(share_cum_df["component"], share_cum_df["share"], label="Share")
            pca_pareto_ax.plot(share_cum_df["component"], np.cumsum(share_cum_df["share"]), color="red", marker="o", label="Cumulative")
            pca_pareto_ax.legend(frameon=False)
            pca_pareto_ax.set_xlabel("Component")
            pca_pareto_ax.set_ylabel("Share of Variance")
            st.pyplot(pca_pareto)
            plt.close()

        with st.expander("Principal Components Loadings"):
            loadings, _ = hl.orient_comps(scaled.components_, scaled.transform(StandardScaler().fit_transform(df[pca_cols])))
            loadings_df = pd.DataFrame(loadings, columns=pca_cols, index=[f"PC{i+1}" for i in range(len(loadings))])
            st.dataframe(loadings_df)

            load_plt, load_ax = plt.subplots(figsize=(10,6))
            sns.heatmap(loadings_df, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, ax=load_ax)
            st.pyplot(load_plt.figure)
            plt.close()
else:
    st.warning("Please upload a CSV file in the sidebar to view PCA results.")