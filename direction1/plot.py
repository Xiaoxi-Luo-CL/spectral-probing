import streamlit as st
import pandas as pd
import os
import json
import numpy as np
from scipy.stats import entropy
from PIL import Image

# Configuration
RESULTS_DIR = "results"

@st.cache_data
def load_data():
    """
    Traverses the results directory to build a summary DataFrame.
    Calculates Entropy immediately upon loading.
    """
    records = []
    
    # 1. Traverse directories
    if not os.path.exists(RESULTS_DIR):
        st.error(f"Directory '{RESULTS_DIR}' not found.")
        return pd.DataFrame()

    for folder_name in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        config_path = os.path.join(folder_path, "config.json")
        dist_path = os.path.join(folder_path, "distribution.npy")
        img_path = os.path.join(folder_path, "spectral_profile.png")
        
        # 2. Parse Config
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                try:
                    conf = json.load(f)
                except:
                    continue
        else:
            continue # Skip if no config

        # 3. Process Distribution (Calculate Entropy)
        dist_entropy = None
        has_dist = False
        if os.path.exists(dist_path):
            try:
                # Load data (assuming 1D probability distribution or logits)
                dist = np.load(dist_path)
                # Ensure it's a probability distribution for entropy
                # (If it's logits, perform softmax first; assuming probs here for simplicity)
                # simple normalization just in case:
                dist_prob = dist / (np.sum(dist) + 1e-9) 
                dist_entropy = entropy(dist_prob)
                has_dist = True
            except Exception as e:
                pass

        # 4. Extract Key Hyperparameters (Customize based on your JSON)
        # Extracting 'task' from 'train_path' loosely
        task_name = conf.get("train_path", "").split("/")[-1].replace(".csv", "")
        
        records.append({
            "id": folder_name,
            "task": task_name,
            "model": conf.get("lm_name", "N/A"),
            "classifier": conf.get("classifier", "N/A"),
            # Check initialization logic (example key mapping)
            "init_method": "frq_scale=" + str(conf.get("frq_scale")) if conf.get("frq_scale") else "default",
            "lr": conf.get("learning_rate"),
            "entropy": dist_entropy,
            "img_path": img_path,
            "dist_path": dist_path,
            "has_dist": has_dist
        })
    from IPython import embed;embed()
    return pd.DataFrame(records)

def main():
    st.set_page_config(layout="wide", page_title="Exp Analysis")
    st.title("Experiment Analysis Dashboard")

    df = load_data()
    
    if df.empty:
        st.warning("No data found.")
        return

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Experiments")
    
    # Dynamic filters based on existing values
    selected_task = st.sidebar.multiselect("Task", df["task"].unique(), default=df["task"].unique())
    selected_model = st.sidebar.multiselect("Model", df["model"].unique(), default=df["model"].unique())
    selected_clf = st.sidebar.multiselect("Classifier", df["classifier"].unique(), default=df["classifier"].unique())

    # Apply filters
    filtered_df = df[
        (df["task"].isin(selected_task)) & 
        (df["model"].isin(selected_model)) &
        (df["classifier"].isin(selected_clf))
    ]

    # --- TABS: Different Analysis Modes ---
    tab1, tab2 = st.tabs(["Overview & Gallery", "Distribution Comparison (KL)"])

    # === TAB 1: Overview ===
    with tab1:
        st.markdown(f"**Total Experiments:** {len(filtered_df)}")
        
        # Display Dataframe with Image Preview and Entropy
        st.dataframe(
            filtered_df[["id", "task", "model", "classifier", "init_method", "entropy", "img_path"]],
            column_config={
                "img_path": st.column_config.ImageColumn("Spectral Profile", help="Preview"),
                "entropy": st.column_config.NumberColumn("Entropy", format="%.4f"),
            },
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("Visual Grid")
        # Grid View for pure visual comparison
        cols = st.columns(4)
        for idx, row in enumerate(filtered_df.itertuples()):
            with cols[idx % 4]:
                if os.path.exists(row.img_path):
                    st.image(row.img_path, use_container_width=True)
                    st.caption(f"**{row.id[:8]}..** | H={row.entropy:.3f}\n{row.model} | {row.classifier}")

    # === TAB 2: Comparison (KL Divergence) ===
    with tab2:
        st.header("Compare Two Settings (KL Divergence)")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            id_a = st.selectbox("Select Baseline (P)", filtered_df["id"].unique(), key="sel_a")
            row_a = df[df["id"] == id_a].iloc[0]
            st.image(row_a["img_path"], caption=f"Baseline: {row_a['classifier']}")
            
        with col_b:
            # Filter out the baseline from the options to avoid self-comparison
            remaining_ids = filtered_df[filtered_df["id"] != id_a]["id"].unique()
            id_b = st.selectbox("Select Target (Q)", remaining_ids, key="sel_b")
            if id_b:
                row_b = df[df["id"] == id_b].iloc[0]
                st.image(row_b["img_path"], caption=f"Target: {row_b['classifier']}")

        # Calculation Section
        if id_a and id_b:
            st.divider()
            
            # Load distributions on demand
            if row_a["has_dist"] and row_b["has_dist"]:
                dist_p = np.load(row_a["dist_path"])
                dist_q = np.load(row_b["dist_path"])
                
                # Normalize just to be safe
                p = dist_p / (np.sum(dist_p) + 1e-9)
                q = dist_q / (np.sum(dist_q) + 1e-9)
                
                # Compute KL Divergence
                kl_val = entropy(p, q)
                
                st.metric(
                    label=f"KL Divergence ( {id_a[:6]} || {id_b[:6]} )",
                    value=f"{kl_val:.6f}",
                    delta=f"Entropy Diff: {row_a['entropy'] - row_b['entropy']:.4f}"
                )
                
                st.info("Note: KL(P || Q) measures how much information is lost when Q is used to approximate P.")
            else:
                st.error("Missing distribution.npy for one of the selected experiments.")

if __name__ == "__main__":
    main()