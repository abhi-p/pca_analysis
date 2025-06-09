import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# App title
st.title("Interactive PCA Analyzer with Loadings Plot")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your data:")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Select output (PCA) variables
    output_vars = st.multiselect("Select output variables for PCA", options=numeric_cols, default=numeric_cols)

    # Select input (grouping/color) variables
    input_vars = st.multiselect(
        "Select input variables for coloring/grouping (e.g. experimental factors)",
        options=[col for col in df.columns if col not in output_vars],
    )
    color_var = st.selectbox("Select input variable to color the PCA scatterplot", options=input_vars if input_vars else [None])

    # Choose PCA components
    n_components = st.radio("Select number of PCA components", options=[2, 3], index=0)

    if len(output_vars) < n_components:
        st.error(f"You need at least {n_components} output variables selected.")
    else:
        # Standardize the data
        X = df[output_vars].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Add PCA results to a new DataFrame
        pca_cols = [f"PC{i+1}" for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        if color_var:
            pca_df[color_var] = df[color_var].values[:len(X_pca)]

        # PCA explained variance
        st.write("### Explained Variance Ratio")
        st.write({f"PC{i+1}": round(var, 3) for i, var in enumerate(pca.explained_variance_ratio_)})

        # PCA Sample Scatter Plot
        # st.write("### PCA Scatterplot")
        # if n_components == 2:
        #     fig1 = plt.figure(figsize=(7, 6))
        #     sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=color_var, s=100, palette="coolwarm")
        #     plt.axhline(0, color='gray', linestyle='--')
        #     plt.axvline(0, color='gray', linestyle='--')
        #     plt.title("PCA Score Plot (Samples)")
        #     plt.grid(True)
        #     st.pyplot(fig1)
        # elif n_components == 3:
        #     from mpl_toolkits.mplot3d import Axes3D
        #     fig2 = plt.figure(figsize=(8, 6))
        #     ax = fig2.add_subplot(111, projection='3d')
        #     scatter = ax.scatter(
        #         X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
        #         c=pd.factorize(df[color_var])[0] if color_var else 'blue',
        #         cmap="coolwarm", s=80
        #     )
        #     ax.set_xlabel("PC1")
        #     ax.set_ylabel("PC2")
        #     ax.set_zlabel("PC3")
        #     ax.set_title("3D PCA Score Plot")
        #     st.pyplot(fig2)

        # SIMCA-style Loadings Plot

        st.write("### Correlation Matrix")


    # Show correlation matrix
        st.write("### Correlation Matrix (Numeric Columns)")

        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        corr = df[num_cols].corr()

        # Format and display using Pandas Styler for background gradient
        st.dataframe(corr.style.background_gradient(cmap='coolwarm').format(precision=2))

        st.write("### SIMCA-style PCA Loadings Plot")
        loadings = pd.DataFrame(pca.components_.T, columns=pca_cols, index=output_vars)

        fig3 = plt.figure(figsize=(7, 7))
        plt.scatter(loadings["PC1"], loadings["PC2"], s=120, color="black")

        for feature, (x, y) in loadings[["PC1", "PC2"]].iterrows():
            plt.text(x, y, feature, fontsize=9, ha='center', va='center', color="blue")

        # Unit circle (optional, only valid if data is standardized)
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
        plt.gca().add_patch(circle)

        plt.axhline(0, color='grey', lw=1)
        plt.axvline(0, color='grey', lw=1)
        plt.xlabel("PC1 Loadings")
        plt.ylabel("PC2 Loadings")
        plt.title("Loadings Plot (Features)")
        plt.grid(True)
        plt.axis('equal')
        st.pyplot(fig3)
