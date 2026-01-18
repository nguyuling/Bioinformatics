"""
Leukemia Diagnosis Application using Streamlit
Name: NGU YU LING
Matric No.: A23CS0149
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
import GEOparse

# configure st page
st.set_page_config(
    page_title="Leukemia Diagnosis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# css
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
    }
    .section-header {
        color: #ff7f0e;
        font-size: 1.8em;
        font-weight: bold;
        margin-top: 30px;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# config
GSE_ID = "GSE13164"
FEATURE = "GSE13164_cleaned_features.csv"
LABEL = "GSE13164_cleaned_labels.csv"
TARGET = ['ALL', 'AML', 'CLL', 'CML']
FEATURE_IDENTIFIER = 'GB_ACC'

#! DATA PROCESSING
@st.cache_data
def download_and_parse_geo(gse_id): 
    with st.spinner(f"Downloading GEO dataset {gse_id}..."):
        try:
            gse = GEOparse.get_GEO(geo=gse_id, destdir="./", silent=True)
            platform_key = list(gse.gpls.keys())[0]
            gpl_table = gse.gpls[platform_key].table.copy()
            st.success(f"Successfully loaded {gse_id}")
            return gse, gpl_table
        except Exception as e:
            st.error(f"Error downloading GEO data: {e}")
            return None, None

@st.cache_data
def extract_and_filter_samples(gse):
    metadata_list = []
    sample_dfs = []
    for name, gsm in gse.gsms.items():
        characteristics = ' '.join(gsm.metadata.get('characteristics_ch1', [''])).lower()
        # leukemia_type
        leukemia_type = next((t for t in TARGET if t.lower() in characteristics), None)
        if not leukemia_type:
            continue
        # id_ref 
        metadata_list.append({'Sample_ID': name, 'Leukemia_Type': leukemia_type})
        gsm_df = gsm.table.copy()
        if 'ID_REF' not in gsm_df.columns:
            continue
        # value_col
        value_col = next((col for col in gsm_df.columns 
                         if col.upper() in ['VALUE', 'LOG_RATIO', 'SIGNAL', 'AVG_SIGNAL']), None)
        if not value_col:
            non_id_cols = [c for c in gsm_df.columns if c != 'ID_REF']
            value_col = non_id_cols[-1] if non_id_cols else None
        if not value_col:
            continue
        # store and append
        gsm_df = gsm_df[['ID_REF', value_col]].rename(columns={value_col: name})
        sample_dfs.append(gsm_df)
    return sample_dfs, pd.DataFrame(metadata_list)

@st.cache_data
def merge_expression_data(sample_dfs):
    expression_data = reduce(lambda left, right: pd.merge(left, right, on='ID_REF', how='inner'), sample_dfs)
    expression_data = expression_data.set_index('ID_REF')
    return expression_data

@st.cache_data
def annotate_and_aggregate_features(expression_data, gpl_table):
    annotation_cols = ['ID', FEATURE_IDENTIFIER]
    annotation_df = gpl_table[annotation_cols].rename(columns={'ID': 'ID_REF', FEATURE_IDENTIFIER: 'Feature_ID'})
    annotation_df.dropna(subset=['Feature_ID'], inplace=True)
    annotation_df = annotation_df[annotation_df['Feature_ID'].str.strip() != '---']
    annotation_df['Feature_ID'] = annotation_df['Feature_ID'].apply(
        lambda x: x.split(' // ')[0].strip()
    )
    merged_df = pd.merge(expression_data.reset_index(), annotation_df, on='ID_REF', how='inner')
    merged_df.dropna(subset=['Feature_ID'], inplace=True)
    sample_cols = [col for col in merged_df.columns if col.startswith('GSM')]
    final_features_df = merged_df.groupby('Feature_ID')[sample_cols].mean()
    final_features_df = final_features_df.T
    return final_features_df

@st.cache_data
def align_and_encode_labels(final_features_df, metadata_df):
    metadata_df = metadata_df.set_index('Sample_ID').loc[final_features_df.index.tolist()].reset_index()
    le = LabelEncoder()
    metadata_df['Target_Code'] = le.fit_transform(metadata_df['Leukemia_Type'])
    return final_features_df, metadata_df, le

@st.cache_data
def load_and_preprocess_geo(gse_id):
    gse, gpl_table = download_and_parse_geo(gse_id)
    if gse is None:
        return None, None, None
    
    sample_dfs, metadata_df = extract_and_filter_samples(gse)
    if not sample_dfs:
        return None, None, None
    
    expression_data = merge_expression_data(sample_dfs)
    final_features_df = annotate_and_aggregate_features(expression_data, gpl_table)
    final_features_df, metadata_df, le = align_and_encode_labels(final_features_df, metadata_df)
    
    return final_features_df, metadata_df, le

@st.cache_data
def load_local_data():
    try:
        features_df = pd.read_csv(FEATURE, index_col=0)
        labels_df = pd.read_csv(LABEL)
        
        # Recreate label encoder
        le = LabelEncoder()
        le.fit(['ALL', 'AML', 'CLL', 'CML'])
        
        return features_df, labels_df, le
    except FileNotFoundError:
        return None, None, None

#! MODEL TRAINING
@st.cache_resource
def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
    models = {}
    predictions = {}
    metrics = {}
    
    # model 1: polynomial (deg=2)
    poly_pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear_regression', Ridge(alpha=1.0))
    ])
    poly_pipeline.fit(X_train_scaled, y_train)
    y_pred_train = poly_pipeline.predict(X_train_scaled)
    y_pred_test = poly_pipeline.predict(X_test_scaled)
    models['Polynomial (deg=2)'] = poly_pipeline
    predictions['Polynomial (deg=2)'] = y_pred_test
    metrics['Polynomial (deg=2)'] = {
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    # model 2: ridge
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train_scaled, y_train)
    y_pred_train = model_ridge.predict(X_train_scaled)
    y_pred_test = model_ridge.predict(X_test_scaled)
    models['Ridge'] = model_ridge
    predictions['Ridge'] = y_pred_test
    metrics['Ridge'] = {
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    # model 3: neural network (MLPRegressor)
    model_nn = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=16,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42,
        verbose=0
    )
    
    model_nn.fit(X_train_scaled, y_train)
    y_pred_train = model_nn.predict(X_train_scaled)
    y_pred_test = model_nn.predict(X_test_scaled)
    models['Neural Network'] = model_nn
    predictions['Neural Network'] = y_pred_test
    metrics['Neural Network'] = {
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    return models, predictions, metrics

#! VISUALIZATION
def plot_class_distribution(labels_df):
    fig = px.bar(
        labels_df['Leukemia_Type'].value_counts().reset_index(),
        x='Leukemia_Type',
        y='count',
        title='Distribution of Leukemia Types',
        labels={'Leukemia_Type': 'Leukemia Type', 'count': 'Number of Samples'},
        color='Leukemia_Type',
        template='plotly_white'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig

def plot_gene_expression_distribution(features_df):
    stats = features_df.describe().T
    fig = px.box(
        features_df.iloc[:, :20],
        title='Gene Expression Distribution (First 20 Genes)',
        template='plotly_white'
    )
    fig.update_layout(height=400)
    return fig

def plot_model_comparison(metrics):
    models_list = list(metrics.keys())
    train_r2 = [metrics[m]['train_r2'] for m in models_list]
    test_r2 = [metrics[m]['test_r2'] for m in models_list]
    train_mse = [metrics[m]['train_mse'] for m in models_list]
    test_mse = [metrics[m]['test_mse'] for m in models_list]
    fig = make_subplots(rows=1, cols=2, subplot_titles=('R² Scores', 'MSE Scores'))
    fig.add_trace(go.Bar(x=models_list, y=train_r2, name='Train R²', marker_color='rgb(31, 119, 180)'), row=1, col=1)
    fig.add_trace(go.Bar(x=models_list, y=test_r2, name='Test R²', marker_color='rgb(158, 202, 225)'), row=1, col=1)
    fig.add_trace(go.Bar(x=models_list, y=train_mse, name='Train MSE', marker_color='rgb(255, 127, 14)'), row=1, col=2)
    fig.add_trace(go.Bar(x=models_list, y=test_mse, name='Test MSE', marker_color='rgb(255, 187, 120)'), row=1, col=2)
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    fig.update_layout(height=400, template='plotly_white', showlegend=True, barmode='group')
    return fig

def plot_predictions_vs_actual(y_test, y_pred, model_name):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(size=8, color='blue', opacity=0.6)
    ))
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{model_name}: Predictions vs Actual Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        height=400,
        template='plotly_white'
    )
    
    return fig

#! MAIN
def main():
    # header
    st.markdown('<h1 class="main-header">Leukemia Diagnosis System</h1>', unsafe_allow_html=True)
    st.markdown("""A machine learning application for classifying leukemia types using gene expression data from the GEO dataset GSE13164.</div>""", unsafe_allow_html=True)
    
    # sidebar
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Select a page:", ["Home", "Data Loading", "Exploratory Analysis", "Model Training", "Predictions & Diagnosis"])
    
    # home page
    if page == "Home":
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                ### About This Application
                This Streamlit application provides a complete pipeline for leukemia diagnosis:
                - **Data Processing**: Load and preprocess GEO dataset GSE13164
                - **EDA**: Explore gene expression patterns
                - **Model Training**: Train and compare multiple ML models
                - **Diagnosis**: Classify leukemia types from gene expression data
                
                ### Leukemia Types
                - **ALL** (Acute Lymphoblastic Leukemia)
                - **AML** (Acute Myeloid Leukemia)
                - **CLL** (Chronic Lymphocytic Leukemia)
                - **CML** (Chronic Myeloid Leukemia)
            """)
        
        with col2:
            st.markdown("""
                ### Features
                - Automatic GEO data download & processing
                - Gene expression analysis
                - Multiple ML model comparison
                - Real-time predictions
                - Interactive visualizations
                
                ### Dataset Information
                - **Source**: GEO (Gene Expression Omnibus)
                - **Accession**: GSE13164
                - **Features**: Gene expression levels
                - **Classes**: 4 leukemia types
            """)
        
        st.info("""
            **Getting Started**: Navigate through the sidebar pages to load data, 
            explore it, train models, and make predictions.
        """)
    
    # loading data page
    elif page == "Data Loading":
        st.markdown('<h2 class="section-header">Data Loading & Preprocessing</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            use_local = st.checkbox("Use local pre-processed data (faster)", value=True)
        
        with col2:
            if st.button("Load Data"):
                if use_local:
                    features_df, labels_df, le = load_local_data()
                    if features_df is not None:
                        st.success("Local data loaded successfully!")
                    else:
                        st.info("No local data found. Attempting to download from GEO...")
                        features_df, labels_df, le = load_and_preprocess_geo(GSE_ID)
                        if features_df is not None:
                            features_df.to_csv(FEATURE)
                            labels_df.to_csv(LABEL, index=False)
                            st.success("Data downloaded and processed!")
                else:
                    features_df, labels_df, le = load_and_preprocess_geo(GSE_ID)
                    if features_df is not None:
                        features_df.to_csv(FEATURE)
                        labels_df.to_csv(LABEL, index=False)
                        st.success("Data downloaded and processed!")
                
                if features_df is not None:
                    st.session_state.features_df = features_df
                    st.session_state.labels_df = labels_df
                    st.session_state.le = le
        
        # show loaded data info
        if 'features_df' in st.session_state:
            st.markdown("### Dataset Summary")
            features_df = st.session_state.features_df
            labels_df = st.session_state.labels_df
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Samples", len(features_df))
            col2.metric("Total Genes", features_df.shape[1])
            col3.metric("Leukemia Types", len(set(labels_df['Leukemia_Type'])))
            col4.metric("Train-Test Split", "80-20")
            
            st.markdown("### Data Preview")
            tab1, tab2 = st.tabs(["Features", "Labels"])
            with tab1:
                st.dataframe(features_df.head(10))
                st.write(f"Shape: {features_df.shape}")
            with tab2:
                st.dataframe(labels_df.head(10))
        else:
            st.warning("Please load data first using the button above.")
    
    # eda page
    elif page == "Exploratory Analysis":
        st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        if 'features_df' not in st.session_state:
            st.error("Please load data first from the 'Data Loading' page.")
            return
        
        features_df = st.session_state.features_df
        labels_df = st.session_state.labels_df
        
        # class
        st.markdown("### Class Distribution")
        fig1 = plot_class_distribution(labels_df)
        st.plotly_chart(fig1, use_container_width=True)
        
        # desc stats
        st.markdown("### Descriptive Statistics")
        st.write("**Overall Gene Expression Statistics**")
        st.dataframe(features_df.describe())
        
        st.write("**Statistics by Leukemia Type**")
        combined_df = labels_df[['Sample_ID', 'Leukemia_Type']].set_index('Sample_ID').join(features_df, how='inner')
        cols = st.columns(4)
        for idx, ltype in enumerate(TARGET):
            with cols[idx]:
                st.write(f"**{ltype}**")
                type_data = combined_df[combined_df['Leukemia_Type'] == ltype].drop('Leukemia_Type', axis=1)
                st.dataframe(type_data.describe().loc[['mean', 'std', 'min', 'max']])
        
        # ge dist
        st.markdown("### Gene Expression Distribution")
        fig2 = plot_gene_expression_distribution(features_df)
        st.plotly_chart(fig2, use_container_width=True)
        
        # corr analysis
        st.markdown("### Correlation Analysis")
        combined_df = labels_df[['Sample_ID', 'Leukemia_Type']].set_index('Sample_ID').join(features_df, how='inner')
        # top 15 genes by variance across entire dataset
        overall_variance = features_df.var().nlargest(15)
        top_genes = overall_variance.index.tolist()
        cols = st.columns(2)
        for idx, ltype in enumerate(TARGET):
            with cols[idx % 2]:
                type_data = combined_df[combined_df['Leukemia_Type'] == ltype].drop('Leukemia_Type', axis=1)
                corr_matrix = type_data[top_genes].corr()
                fig = go.Figure(data=go.Heatmap(z=corr_matrix.values[::-1], x=top_genes, y=top_genes[::-1], colorscale='RdBu_r', zmin=-1, zmax=1))
                fig.update_layout(title=f'{ltype} (Top 15 Genes)', height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    # model training page
    elif page == "Model Training":
        st.markdown('<h2 class="section-header">Model Training & Comparison</h2>', unsafe_allow_html=True)
        if 'features_df' not in st.session_state:
            st.error("Please load data first from the 'Data Loading' page.")
            return
        
        features_df = st.session_state.features_df
        labels_df = st.session_state.labels_df
        le = st.session_state.le
        
        X = features_df.values
        y = labels_df['Target_Code'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Train All Models"):
                with st.spinner("Training models. This may take a moment..."):
                    models, predictions, metrics = train_models(
                        X_train_scaled, X_test_scaled, y_train, y_test
                    )
                    st.session_state.models = models
                    st.session_state.predictions = predictions
                    st.session_state.metrics = metrics
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.scaler = scaler
                    st.success("Model training completed!")
        with col2:
            if st.button("Clear Cache & Retrain"):
                st.cache_resource.clear()
                st.rerun()
        
        if 'metrics' in st.session_state:
            st.markdown("### Model Performance Comparison")
            metrics = st.session_state.metrics
            comparison_data = []
            for model_name, metric in metrics.items():
                comparison_data.append({
                    'Model': model_name,
                    'Train R²': metric['train_r2'],
                    'Test R²': metric['test_r2'],
                    'Train MSE': metric['train_mse'],
                    'Test MSE': metric['test_mse']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # visualization
            fig = plot_model_comparison(metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # best model
            best_r2_idx = comparison_df['Test R²'].idxmax()
            best_mse_idx = comparison_df['Test MSE'].idxmin()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Test R²", 
                         comparison_df.loc[best_r2_idx, 'Model'],
                         f"{comparison_df.loc[best_r2_idx, 'Test R²']:.4f}")
            with col2:
                st.metric("Best Test MSE",
                         comparison_df.loc[best_mse_idx, 'Model'],
                         f"{comparison_df.loc[best_mse_idx, 'Test MSE']:.4f}")
            
            # models analysis
            st.markdown("### Individual Model Analysis")
            y_test = st.session_state.y_test
            cols = st.columns(2)
            for idx, selected_model in enumerate(st.session_state.metrics.keys()):
                with cols[idx % 2]:
                    y_pred = st.session_state.predictions[selected_model]
                    fig = plot_predictions_vs_actual(y_test, y_pred, selected_model)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click 'Train All Models' to start training.")
    
    # predictions page
    elif page == "Predictions & Diagnosis":
        st.markdown('<h2 class="section-header">Leukemia Diagnosis & Predictions</h2>', unsafe_allow_html=True)
        if 'models' not in st.session_state:
            st.error("Please train models first from the 'Model Training' page.")
            return
        
        models = st.session_state.models
        y_test = st.session_state.y_test
        scaler = st.session_state.scaler
        selected_model = st.selectbox("Select the model for diagnosis:", list(models.keys()))
        st.markdown(f"### Using {selected_model} Model")
        
        # classification result
        model = models[selected_model]
        X_test = st.session_state.X_test
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_class = np.clip(np.round(y_pred), 0, 3).astype(int)
        class_to_type = {0: 'ALL', 1: 'AML', 2: 'CLL', 3: 'CML'}
        y_test_types = [class_to_type[y] for y in y_test]
        y_pred_types = [class_to_type[y] for y in y_pred_class]
        
        # metrics
        accuracy = accuracy_score(y_test, y_pred_class)
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Accuracy", f"{accuracy:.2%}")
        col2.metric("Samples Tested", len(y_test))
        col3.metric("Correct Predictions", int(accuracy * len(y_test)))
        
        # confusion matrix
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=['ALL', 'AML', 'CLL', 'CML'],
            y=['ALL', 'AML', 'CLL', 'CML'],
            text=cm,
            texttemplate="%{text}",
            colorscale='Blues'
        ))
        fig.update_layout(
            title=f'Confusion Matrix - {selected_model}',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # classification report
        st.markdown("### Classification Report")
        report = classification_report(y_test, y_pred_class, 
                                      target_names=['ALL', 'AML', 'CLL', 'CML'],
                                      output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        # sample predictions
        st.markdown("### Sample Predictions")
        predictions_df = pd.DataFrame({
            'Sample Index': range(len(y_test)),
            'Actual Type': y_test_types,
            'Predicted Type': y_pred_types,
            'Continuous Prediction': y_pred,
            'Correct': [y_true == y_pred for y_true, y_pred in zip(y_test_types, y_pred_types)]
        })
        
        # filter options
        col1, col2 = st.columns(2)
        with col1:
            show_all = st.checkbox("Show all predictions", value=False)
        with col2:
            show_errors = st.checkbox("Show only incorrect predictions", value=False)
        display_df = predictions_df
        if show_errors:
            display_df = display_df[~display_df['Correct']]
        if show_all:
            st.dataframe(display_df, use_container_width=True)
        else:
            st.dataframe(display_df.head(20), use_container_width=True)
        
        # accuracy by type
        st.markdown("### Accuracy by Leukemia Type")
        accuracy_by_type = []
        for ltype in ['ALL', 'AML', 'CLL', 'CML']:
            mask = predictions_df['Actual Type'] == ltype
            type_acc = predictions_df[mask]['Correct'].sum() / mask.sum() if mask.sum() > 0 else 0
            accuracy_by_type.append({
                'Leukemia Type': ltype,
                'Accuracy': type_acc,
                'Samples': mask.sum()
            })
        type_acc_df = pd.DataFrame(accuracy_by_type)
        fig = px.bar(type_acc_df, x='Leukemia Type', y='Accuracy',
                    title='Classification Accuracy by Leukemia Type',
                    color='Accuracy',
                    labels={'Accuracy': 'Accuracy Rate'},
                    template='plotly_white')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()