import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="DeepFake Detection Research Project",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Custom CSS for deep blue theme
st.markdown("""
    <style>
        body {
            background-color: #010d1a;
        }

        /* Button styling */
        div.stButton > button {
            background-color: #002855;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5em 1em;
            font-weight: bold;
        }

        div.stButton > button:hover {
            background-color: #004080;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #4CAF50;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .team-card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .methodology-section {
        padding: 1rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        background: rgba(76, 175, 80, 0.1);
        border-radius: 0 8px 8px 0;
    }
    
    .nav-tabs {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .nav-tab {
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border: 1px solid #ccc;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .nav-tab:hover {
        background-color: #4CAF50;
        color: white;
    }
    
    .highlight-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
        background: rgba(33, 150, 243, 0.1);
    }
    
    .cross-dataset-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #FF5722;
        background: rgba(255, 87, 34, 0.1);
    }
    
    .same-dataset-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
        background: rgba(76, 175, 80, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Navigation menu
def create_navigation():
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
    with col2:
        if st.button("👥 Team", use_container_width=True):
            st.session_state.page = "team"
    with col3:
        if st.button("🔬 Methodology", use_container_width=True):
            st.session_state.page = "methodology"
    with col4:
        
        if st.button("📈 Comparisons", use_container_width=True):
            st.session_state.page = "comparisons"    
    with col5:
        if st.button("📊 Results", use_container_width=True):
            st.session_state.page = "results"
    with col6:
        if st.button("🎥 Demo", use_container_width=True):
            st.session_state.page = "demo"
    with col7:
        if st.button("📄 Resources", use_container_width=True):
            st.session_state.page = "resources"

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔍 DeepFake Detection Research Project</h1>
    <p><em>Advanced CNN, Hybrid and Transformer Models for Authentic Media Verification</em></p>
</div>
""", unsafe_allow_html=True)

# Navigation
create_navigation()

# Data for visualizations - Same Dataset (FaceForensics++)
same_dataset_data = {
    'Model': ['XceptionNet (30F)', 'XceptionNet (5F)', 'EfficientNet (5F)', 
              'Hybrid Model (5F)', 'ResNet50 (5F)', 'r3d_18 (10F)', 'Slow ResNet (10F)'],
    'AUC Score': [0.9880, 0.96, 0.96, 0.96, 0.91, 0.89, 0.89],
    'Accuracy (%)': [95.72, 93, 92, 93, 89, 89, 89],
    'Type': ['2D CNN', '2D CNN', '2D CNN', 'Hybrid', '2D CNN', '3D CNN', '3D CNN'],
    'Frames': [30, 5, 5, 5, 5, 10, 10],
    'Dataset': ['Same'] * 7
}

# Data for visualizations - Cross Dataset (FaceForensics++ → Celeb-DF)
cross_dataset_data = {
    'Model': ['Swin-S', 'ViT-16', 'ConvNeXt-S', 'Xception', '3D ResNet', 
              '3D ResNet + Transformer', 'Xception + Transformer'],
    'AUC Score': [0.8533, 0.8512, 0.8255, 0.8041, 0.7923, 0.7521, 0.7489],
    'Accuracy (%)': [78.35, 78.49, 76.92, 71.88, 81.16, 79.02, 63.90],
    'Type': ['Vision Transformer', 'Vision Transformer', 'CNN', '2D CNN', '3D CNN', 
             'Hybrid', 'Hybrid'],
    'Frames': [30, 30, 30, 30, 30, 30, 30],
    'Dataset': ['Cross'] * 7
}

# Combined dataset for overall comparison
combined_data = {
    'Model': ['XceptionNet (30F) (Same)', 'XceptionNet (5F) (Same)', 'EfficientNet (5F) (Same)', 
              'Hybrid Model (5F) (Same)', 'ResNet50 (5F) (Same)', 'r3d_18 (10F) (Same)', 
              'Slow ResNet (10F) (Same)', 'Swin-S (Cross)', 'ViT-16 (Cross)', 
              'ConvNeXt-S (Cross)', 'Xception (Cross)', '3D ResNet (Cross)', 
              '3D ResNet + Transformer (Cross)', 'Xception + Transformer (Cross)'],
    'AUC Score': [0.9880, 0.96, 0.96, 0.96, 0.91, 0.89, 0.89, 0.8533, 0.8512, 
                  0.8255, 0.8041, 0.7923, 0.7521, 0.7489],
    'Accuracy (%)': [95.72, 93, 92, 93, 89, 89, 89, 78.35, 78.49, 76.92, 
                     71.88, 81.16, 79.02, 63.90],
    'Type': ['2D CNN', '2D CNN', '2D CNN', 'Hybrid', '2D CNN', '3D CNN', '3D CNN',
             'Vision Transformer', 'Vision Transformer', 'CNN', '2D CNN', '3D CNN', 
             'Hybrid', 'Hybrid'],
    'Dataset': ['Same', 'Same', 'Same', 'Same', 'Same', 'Same', 'Same',
                'Cross', 'Cross', 'Cross', 'Cross', 'Cross', 'Cross', 'Cross']
}

# DeepFake methods performance
methods_data = {
    'Method': ['FaceSwap', 'Deepfakes', 'Face2Face', 'NeuralTextures'],
    'AUC Score': [0.9921, 0.9937, 0.9877, 0.9785]
}

# Perturbation results
perturbation_data = {
    'Model': ['Slow-R50', 'ViT-16', 'Swin-S', 'ConvNeXt-S'],
    'Downsample_200': [0.5656, 0.6623, 0.6294, 0.6374],
    'Blur_0.5': [0.5668, 0.6669, 0.6784, 0.6715],
    'Sharpen_0.5': [0.5674, 0.6728, 0.6742, 0.6939]
}

# Page content
if st.session_state.page == "home":
    st.markdown("## 🎯 Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🏆 Same-Dataset</h3>
            <h2>XceptionNet</h2>
            <p>AUC: 0.9880</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🏆 Cross-Dataset</h3>
            <h2>Swin-S</h2>
            <p>AUC: 0.8533</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Total Models</h3>
            <h2>14</h2>
            <p>Architectures Tested</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🎥 Dataset Size</h3>
            <h2>5,000</h2>
            <p>Videos Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 📋 Abstract")
    st.markdown("""
    <div class="highlight-box">
    <p>This research project addresses the critical challenge of DeepFake detection in digital media. 
    We evaluated multiple state-of-the-art architectures including ResNet50, EfficientNet, XceptionNet, 
    and advanced Vision Transformers like Swin-S and ViT-16. Our comprehensive analysis spans 2D CNNs, 
    3D CNNs, and hybrid models, with extensive experimentation on both same-dataset and cross-dataset evaluations.</p>
    
    <p><strong>Key Achievements:</strong></p>
    <ul>
        <li>🏆 XceptionNet achieved 95.72% accuracy and 0.9880 AUC on same-dataset evaluation</li>
        <li>🌐 Swin-S achieved the highest cross-dataset AUC score of 0.8533</li>
        <li>📈 Comprehensive comparison of 14 different model configurations</li>
        <li>📊 Detailed analysis of 4 DeepFake generation methods</li>
        <li>🔬 Cross-dataset generalization study (FaceForensics++ → Celeb-DF)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "team":
    st.markdown("## 👥 Research Team")
    
    col0, col1, col2, col3 = st.columns(4)
    
    with col0:
        st.markdown("""
        <div class="team-card">
            <h3>Prof. Nazlı İkizler Cinbiş</h3>
            <p>Project Advisor</p>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        st.markdown("""
        <div class="team-card">
            <h3>👩‍💻 Sena Yalçın</h3>
            <p><strong>Student ID:</strong> 2200356049</p>
            <p>Computer/AI Engineering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="team-card">
            <h3>👩‍💻 İ. Zeynep Karakaş</h3>
            <p><strong>Student ID:</strong> 2200765027</p>
            <p>Computer/AI Engineering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="team-card">
            <h3>👨‍💻 Emre Büyükyılmaz</h3>
            <p><strong>Student ID:</strong> 2220765049</p>
            <p>Computer/AI Engineering</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 🏛️ Institution")
    st.markdown("""
    <div class="highlight-box">
    <h3>Hacettepe University</h3>
    <p><strong>Department:</strong> Computer/Artificial Intelligence Engineering</p>
    <p><strong>Course:</strong> BBM480 Design Project - 2025 Spring</p>
    <p><strong>Supervisor:</strong> Faculty of Engineering</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "methodology":
    st.markdown("## 🔬 Research Methodology")
    
    st.markdown("""
    <div class="methodology-section">
    <h3>📊 Dataset Preparation</h3>
    <p><strong>Training:</strong> FaceForensics++ Dataset</p>
    <p><strong>Cross-Evaluation:</strong> Celeb-DF Dataset</p>
    <ul>
        <li>1,000 original videos + 4,000 DeepFake videos (4 methods × 1,000 each)</li>
        <li>Frame extraction: 5, 10, and 30 frames per video tested</li>
        <li>Face detection and cropping for focused analysis</li>
        <li>Preprocessing: Resize (224×224 or 299×299), normalization, augmentation</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="methodology-section">
        <h3>🧠 Model Architectures</h3>
        <ul>
            <li><strong>2D CNNs:</strong> ResNet50, EfficientNet, XceptionNet</li>
            <li><strong>3D CNNs:</strong> r3d_18, Slow ResNet, Custom 3D ResNet</li>
            <li><strong>Vision Transformers:</strong> ViT-16, Swin-S, ConvNeXt-S</li>
            <li><strong>Hybrid Models:</strong> Ensemble approaches</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="methodology-section">
        <h3>⚙️ Training Configuration</h3>
        <ul>
            <li><strong>Learning Rate:</strong> 0.002</li>
            <li><strong>Optimizer:</strong> Adam</li>
            <li><strong>Loss Function:</strong> Cross-Entropy</li>
            <li><strong>Pre-training:</strong> ImageNet weights</li>
            <li><strong>Class Weighting:</strong> 5:1 for imbalanced data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="methodology-section">
    <h3>📏 Evaluation Strategy</h3>
    <p><strong>Same-Dataset Evaluation:</strong> Training and testing on FaceForensics++</p>
    <p><strong>Cross-Dataset Evaluation:</strong> Training on FaceForensics++, testing on Celeb-DF</p>
    <ul>
        <li><strong>ROC AUC Score:</strong> Primary metric for model comparison</li>
        <li><strong>Accuracy:</strong> Secondary metric</li>
        <li><strong>Confusion Matrix:</strong> For detailed error analysis</li>
        <li><strong>Per-method Analysis:</strong> Performance on individual DeepFake techniques</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "results":
    st.markdown("## 📊 Experimental Results")
        # Dataset comparison summary
    st.markdown("### 📊 Dataset Evaluation Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="same-dataset-box">
        <h4>🎯 Same-Dataset (FaceForensics++)</h4>
        <ul>
            <li><strong>Best Model:</strong> XceptionNet (30F)</li>
            <li><strong>Best AUC:</strong> 0.9880</li>
            <li><strong>Best Accuracy:</strong> 95.72%</li>
            <li><strong>Advantage:</strong> High performance on familiar data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="cross-dataset-box">
        <h4>🌐 Cross-Dataset (FF++ → Celeb-DF)</h4>
        <ul>
            <li><strong>Best Model:</strong> Swin-S</li>
            <li><strong>Best AUC:</strong> 0.8533</li>
            <li><strong>Best Accuracy:</strong> 78.35%</li>
            <li><strong>Advantage:</strong> Better generalization capability</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Individual model analysis
    st.markdown("### 🔍 Detailed Model Analysis")
    
    analysis_tabs = st.tabs(["Same-Dataset Champions", "Cross-Dataset Champions", "Architecture Insights"])
    
    with analysis_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **XceptionNet (30F) - Same Dataset Leader:**
            - AUC Score: 0.9880
            - Accuracy: 95.72%
            - Excellent at detecting subtle facial inconsistencies
            - Superior boundary detection capabilities
            - Benefits from more temporal information (30 frames)
            """)
        with col2:
            st.markdown("""
            **EfficientNet & Hybrid (5F) Performance:**
            - Both achieved AUC: 0.96
            - Accuracy: 92-93%
            - Optimal computational efficiency
            - Good balance between performance and resources
            - Effective with limited frames
            """)
    
    with analysis_tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Swin-S - Cross Dataset Leader:**
            - AUC Score: 0.8533
            - Accuracy: 78.35%
            - **Best generalization capability**
            - Hierarchical feature learning
            - Robust to domain shift
            """)
        with col2:
            st.markdown("""
            **ViT-16 - Strong Cross-Dataset Performance:**
            - AUC Score: 0.8512
            - Accuracy: 78.49%
            - Strong attention mechanisms
            - Good cross-dataset adaptability
            - Vision Transformer advantages
            """)
    
    with analysis_tabs[2]:
        st.markdown("""
        **Key Insights:**
        
        🔸 **Same-Dataset vs Cross-Dataset Performance Gap:**
        - Same-dataset: Up to 98.80% AUC
        - Cross-dataset: Up to 85.33% AUC
        - **~13% performance drop** indicates domain shift challenges
        
        🔸 **Architecture Preferences:**
        - **Same-Dataset:** Traditional CNNs (XceptionNet) excel
        - **Cross-Dataset:** Vision Transformers (Swin-S, ViT-16) lead
        
        🔸 **Generalization Capability:**
        - Vision Transformers show better cross-domain generalization
        - Traditional CNNs may overfit to training data characteristics
        """)
    # DeepFake Methods Performance
    st.markdown("### 🎭 Performance by DeepFake Method")
    
    methods_df = pd.DataFrame(methods_data)
    fig_methods = px.bar(
        methods_df, 
        x='Method', 
        y='AUC Score',
        title='AUC Score by DeepFake Generation Method (Same-Dataset)',
        color='AUC Score',
        color_continuous_scale='viridis'
    )
    fig_methods.update_layout(
        height=400,
        yaxis=dict(range=[0.9, 1.0])  
    )
    st.plotly_chart(fig_methods, use_container_width=True)
    
    # Robustness Analysis
    st.markdown("### 🔍 Robustness Analysis")

    # Real perturbation data from your tables
    perturbation_data = {
        'Model': ['Slow-R50', 'ViT-16', 'Swin-S', 'ConvNeXt-S'],
        # Original performance (baseline - using best performance from tables as reference)
        'Original': [0.5674, 0.6738, 0.6803, 0.6939],
        # Downsample perturbations (using 200 size as primary)
        'Downsample_200': [0.5656, 0.6623, 0.6294, 0.6374],
        'Downsample_168': [0.5661, 0.6434, 0.5756, 0.5849],
        'Downsample_112': [0.5637, 0.5673, 0.5598, 0.5639],
        # Blur perturbations
        'Blur_0.2': [0.5667, 0.6508, 0.6803, 0.6811],
        'Blur_0.5': [0.5668, 0.6669, 0.6784, 0.6715],
        'Blur_2.0': [0.5547, 0.5585, 0.5401, 0.5371],
        # Sharpen perturbations
        'Sharpen_0.2': [0.5673, 0.6738, 0.6602, 0.6858],
        'Sharpen_0.5': [0.5674, 0.6728, 0.6742, 0.6939],
        'Sharpen_2.0': [0.5610, 0.6195, 0.6604, 0.6468],
        # Salt & Pepper noise perturbations
        'SaltPepper_0.001': [0.5642, 0.6374, 0.6647, 0.6773],
        'SaltPepper_0.01': [0.5545, 0.6000, 0.6083, 0.6406],
        'SaltPepper_0.03': [0.5494, 0.5726, 0.5600, 0.6018],
        # Runtime data (converted to minutes)
        'Runtime_min': [11.48, 25.33, 24.08, 22.35]
    }

    pert_df = pd.DataFrame(perturbation_data)

    # Create tabs for different perturbation types
    tab1, tab2, tab3, tab4 = st.tabs(["📐 Downsample", "🌫️ Blur", "✨ Sharpen", "🔘 Salt & Pepper"])

    with tab1:
        st.markdown("#### Downsample-Upsample Robustness")
        fig_downsample = go.Figure()
        downsample_cols = ['Downsample_200', 'Downsample_168', 'Downsample_112']
        colors_down = ['#FF6B6B', '#FF8E8E', '#FFB1B1']
    
        for i, column in enumerate(downsample_cols):
            size = column.split('_')[1]
            fig_downsample.add_trace(go.Bar(
                name=f'Size {size}px',
                x=pert_df['Model'],
                y=pert_df[column],
                text=pert_df[column].round(4),
                textposition='auto',
                marker_color=colors_down[i]
            ))
    
        fig_downsample.update_layout(
            title='Model Performance Under Different Downsample Sizes',
            xaxis_title='Model Architecture',
            yaxis_title='AUC Score',
            barmode='group',
            height=500,
            yaxis=dict(range=[0.5, 0.75])
        )
        st.plotly_chart(fig_downsample, use_container_width=True)

    with tab2:
        st.markdown("#### Blur Robustness")
        fig_blur = go.Figure()
        blur_cols = ['Blur_0.2', 'Blur_0.5', 'Blur_2.0']
        colors_blur = ['#4ECDC4', '#5ED3D3', '#6EEAEA']
    
        for i, column in enumerate(blur_cols):
            sigma = column.split('_')[1]
            fig_blur.add_trace(go.Bar(
                name=f'Sigma {sigma}',
                x=pert_df['Model'],
                y=pert_df[column],
                text=pert_df[column].round(4),
                textposition='auto',
                marker_color=colors_blur[i]
            ))
    
        fig_blur.update_layout(
            title='Model Performance Under Different Blur Levels',
            xaxis_title='Model Architecture',
            yaxis_title='AUC Score',
            barmode='group',
            height=500,
            yaxis=dict(range=[0.5, 0.75])
        )
        st.plotly_chart(fig_blur, use_container_width=True)

    with tab3:
        st.markdown("#### Sharpen Robustness")
        fig_sharpen = go.Figure()
        sharpen_cols = ['Sharpen_0.2', 'Sharpen_0.5', 'Sharpen_2.0']
        colors_sharpen = ['#45B7D1', '#67C7E8', '#89D7EF']
    
        for i, column in enumerate(sharpen_cols):
            sigma = column.split('_')[1]
            fig_sharpen.add_trace(go.Bar(
                name=f'Sigma {sigma}',
                x=pert_df['Model'],
                y=pert_df[column],
                text=pert_df[column].round(4),
                textposition='auto',
                marker_color=colors_sharpen[i]
            ))
    
        fig_sharpen.update_layout(
            title='Model Performance Under Different Sharpen Levels',
            xaxis_title='Model Architecture',
            yaxis_title='AUC Score',
            barmode='group',
            height=500,
            yaxis=dict(range=[0.5, 0.75])
        )
        st.plotly_chart(fig_sharpen, use_container_width=True)

    with tab4:
        st.markdown("#### Salt & Pepper Noise Robustness")
        fig_noise = go.Figure()
        noise_cols = ['SaltPepper_0.001', 'SaltPepper_0.01', 'SaltPepper_0.03']
        colors_noise = ['#96CEB4', '#A8D6C4', '#BADED4']
    
        for i, column in enumerate(noise_cols):
            prob = column.split('_')[1]
            fig_noise.add_trace(go.Bar(
                name=f'Prob {prob}',
                x=pert_df['Model'],
                y=pert_df[column],
                text=pert_df[column].round(4),
                textposition='auto',
                marker_color=colors_noise[i]
            ))
    
        fig_noise.update_layout(
            title='Model Performance Under Different Salt & Pepper Noise Levels',
            xaxis_title='Model Architecture',
            yaxis_title='AUC Score',
            barmode='group',
            height=500,
            yaxis=dict(range=[0.5, 0.75])
        )
        st.plotly_chart(fig_noise, use_container_width=True)

    # Overall robustness analysis
    st.markdown("#### 📊 Overall Robustness Summary")

    col1, col2 = st.columns(2)

    with col1:
        # Calculate average robustness across all perturbations
        st.markdown("#### Robustness Rankings")
        robustness_data = []
    
        for i, model in enumerate(pert_df['Model']):
            # Calculate average performance across moderate perturbations
            moderate_perturbs = [
                pert_df.iloc[i]['Downsample_200'],
                pert_df.iloc[i]['Blur_0.5'],
                pert_df.iloc[i]['Sharpen_0.5'],
                pert_df.iloc[i]['SaltPepper_0.01']
            ]
            avg_robust = np.mean(moderate_perturbs)
            retention = (avg_robust / pert_df.iloc[i]['Original']) * 100
        
            robustness_data.append({
                'Model': model,
                'Avg AUC': f"{avg_robust:.4f}",
                'Runtime': f"{pert_df.iloc[i]['Runtime_min']:.1f}m"
            })
    
        robust_df = pd.DataFrame(robustness_data)
        # Sort by average AUC
        robust_df['Avg_AUC_float'] = robust_df['Avg AUC'].str.replace('f', '').astype(float)
        robust_df = robust_df.sort_values('Avg_AUC_float', ascending=False).drop('Avg_AUC_float', axis=1)
    
        st.dataframe(robust_df, use_container_width=True)

    with col2:
        # Performance vs Runtime scatter plot
        st.markdown("#### Performance vs Runtime Trade-off")
        fig_scatter = go.Figure()
    
        # Use original performance for scatter plot
        fig_scatter.add_trace(go.Scatter(
            x=pert_df['Runtime_min'],
            y=pert_df['Original'],
            mode='markers+text',
            text=pert_df['Model'],
            textposition="top center",
            marker=dict(
                size=15,
                color=pert_df['Original'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="AUC Score")
            ),
            hovertemplate='<b>%{text}</b><br>Runtime: %{x:.1f}m<br>AUC: %{y:.4f}<extra></extra>'
        ))
    
        fig_scatter.update_layout(
            xaxis_title='Inference Runtime (minutes)',
            yaxis_title='Best AUC Score',
            height=400,
            showlegend=False
        )
    
        st.plotly_chart(fig_scatter, use_container_width=True)



    # Key insights based on actual data
    st.markdown("""
    <div class="highlight-box">
    <h4>🔬 Key Robustness Insights from Analysis</h4>
    <ul>
        <li><strong>ConvNeXt-S</strong> shows the best overall performance (0.6939 AUC) with good robustness across perturbations</li>
        <li><strong>Swin-S</strong> demonstrates excellent blur robustness, maintaining performance even at σ=0.5 blur</li>
        <li><strong>ViT-16</strong> shows strong sharpen robustness but is more sensitive to salt & pepper noise</li>
        <li><strong>Slow-R50</strong> offers fastest training (11.5m) but significantly lower performance across all metrics</li>
        <li><strong>Severe perturbations</strong> (Downsample to 112px, Blur σ=2.0, S&P 0.03) cause major performance drops for all models</li>
        <li><strong>Clinical Recommendation:</strong> ConvNeXt-S for best accuracy-robustness balance, Slow-R50 only for rapid prototyping</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)






elif st.session_state.page == "comparisons":
    st.markdown("## 📈 Comprehensive Model Comparisons")
    
    # Evaluation type selection
    comparison_tabs = st.tabs(["🎯 Same-Dataset Evaluation", "🌐 Cross-Dataset Evaluation", "📊 Overall Comparison", "📈 Detection Confidence Simulation"])
    
    with comparison_tabs[0]:
        st.markdown("### 🎯 Same-Dataset Performance (FaceForensics++)")
        st.markdown("""
        <div class="same-dataset-box">
        <p><strong>Training:</strong> FaceForensics++ | <strong>Testing:</strong> FaceForensics++</p>
        <p>This evaluation shows how well models perform on the same type of data they were trained on.</p>
        </div>
        """, unsafe_allow_html=True)
        
        df_same = pd.DataFrame(same_dataset_data)
        
        # Same-dataset AUC comparison
        fig_same_auc = px.bar(
            df_same.sort_values('AUC Score', ascending=True),
            x='AUC Score',
            y='Model',
            color='Type',
            orientation='h',
            title='Same-Dataset AUC Score Comparison',
            color_discrete_map={
                '2D CNN': '#4ECDC4',
                '3D CNN': '#45B7D1',
                'Hybrid': '#96CEB4'
            }
        )
        fig_same_auc.update_layout(height=500)
        st.plotly_chart(fig_same_auc, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy vs AUC scatter
            fig_scatter_same = px.scatter(
                df_same,
                x='Accuracy (%)',
                y='AUC Score',
                size='Frames',
                color='Type',
                hover_data=['Model'],
                title='Accuracy vs AUC (Same-Dataset)'
            )
            st.plotly_chart(fig_scatter_same, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Same-Dataset Rankings")
            same_ranking = df_same.sort_values('AUC Score', ascending=False)[['Model', 'AUC Score', 'Accuracy (%)', 'Type']]
            st.dataframe(same_ranking, hide_index=True, use_container_width=True)
    
    with comparison_tabs[1]:
        st.markdown("### 🌐 Cross-Dataset Performance (FaceForensics++ → Celeb-DF)")
        st.markdown("""
        <div class="cross-dataset-box">
        <p><strong>Training:</strong> FaceForensics++ | <strong>Testing:</strong> Celeb-DF</p>
        <p>This evaluation measures generalization capability across different datasets and DeepFake generation methods.</p>
        </div>
        """, unsafe_allow_html=True)
        
        df_cross = pd.DataFrame(cross_dataset_data)
        
        # Cross-dataset AUC comparison
        fig_cross_auc = px.bar(
            df_cross.sort_values('AUC Score', ascending=True),
            x='AUC Score',
            y='Model',
            color='Type',
            orientation='h',
            title='Cross-Dataset AUC Score Comparison',
            color_discrete_map={
                'Vision Transformer': '#FF6B6B',
                '2D CNN': '#4ECDC4',
                '3D CNN': '#45B7D1',
                'Hybrid': '#96CEB4',
                'CNN': '#FECA57'
            }
        )
        fig_cross_auc.update_layout(height=500)
        st.plotly_chart(fig_cross_auc, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy vs AUC scatter for cross-dataset
            fig_scatter_cross = px.scatter(
                df_cross,
                x='Accuracy (%)',
                y='AUC Score',
                color='Type',
                hover_data=['Model'],
                title='Accuracy vs AUC (Cross-Dataset)',
                size_max=15
            )
            fig_scatter_cross.update_layout(height=400)
            st.plotly_chart(fig_scatter_cross, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Cross-Dataset Rankings")
            cross_ranking = df_cross.sort_values('AUC Score', ascending=False)[['Model', 'AUC Score', 'Accuracy (%)', 'Type']]
            st.dataframe(cross_ranking, hide_index=True, use_container_width=True)
    
    with comparison_tabs[2]:
        st.markdown("### 📊 Overall Model Comparison")
        st.markdown("""
        <div class="highlight-box">
        <p>Complete comparison including both same-dataset and cross-dataset evaluations. 
        Models are labeled with their evaluation type for clarity.</p>
        </div>
        """, unsafe_allow_html=True)
        
        df_combined = pd.DataFrame(combined_data)
        
        # Combined comparison chart
        fig_combined = px.scatter(
            df_combined,
            x='Accuracy (%)',
            y='AUC Score',
            color='Dataset',
            size=[15] * len(df_combined),
            hover_data=['Model', 'Type'],
            title='Complete Model Performance Comparison',
            color_discrete_map={
                'Same': '#4CAF50',
                'Cross': '#FF5722'
            }
        )
        fig_combined.update_layout(height=600)
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # Performance gap analysis
        st.markdown("### 📉 Performance Gap Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Same-Dataset</h3>
                <h2>98.80%</h2>
                <p>Best AUC Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🌐 Cross-Dataset</h3>
                <h2>85.33%</h2>
                <p>Best AUC Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            gap = (0.9880 - 0.8533) / 0.9880 * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>📉 Performance Gap</h3>
                <h2>{gap:.1f}%</h2>
                <p>Generalization Challenge</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Architecture type performance summary
        st.markdown("### 🏗️ Architecture Performance Summary")
        
        same_df = pd.DataFrame(same_dataset_data)
        cross_df = pd.DataFrame(cross_dataset_data)
        
        # Calculate averages by type
        same_type_avg = same_df.groupby('Type')['AUC Score'].agg(['mean', 'max']).round(4)
        cross_type_avg = cross_df.groupby('Type')['AUC Score'].agg(['mean', 'max']).round(4)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Same-Dataset by Architecture")
            same_type_avg.columns = ['Avg AUC', 'Max AUC']
            st.dataframe(same_type_avg, use_container_width=True)

        with col2:
            st.markdown("#### 🌐 Cross-Dataset by Architecture")
            cross_type_avg.columns = ['Avg AUC', 'Max AUC']
            st.dataframe(cross_type_avg, use_container_width=True)

    with comparison_tabs[3]:
        # Visualization of confidence over time
        st.markdown("### 📈 Detection Confidence Simulation")
            # Model selection
        col1, col2 = st.columns(2)
    
        with col1:
            selected_model = st.selectbox(
                "Select Model Architecture:",
                ["XceptionNet (30F)", "Swin-S", "ViT-16", "EfficientNet (5F)", 
                "3D ResNet", "Hybrid Model (5F)"]
            )
        
            evaluation_type = st.selectbox(
                "Evaluation Type:",
                ["Same-Dataset (FaceForensics++)", "Cross-Dataset (Celeb-DF)"]
            )
    
        with col2:
            deepfake_method = st.selectbox(
                "DeepFake Generation Method:",
                ["FaceSwap", "Deepfakes", "Face2Face", "NeuralTextures"]
            )
        
            video_quality = st.selectbox(
                "Video Quality:",
                ["High Quality", "Compressed", "Low Resolution"]
            )
    
        # Simulate prediction
        col1, col2, col3 = st.columns(3)
    
        # Get model performance based on selection
        model_performances = {
            "XceptionNet (30F)": {"same": 0.9880, "cross": 0.8041},
            "Swin-S": {"same": 0.85, "cross": 0.8533},
            "ViT-16": {"same": 0.83, "cross": 0.8512},
            "EfficientNet (5F)": {"same": 0.96, "cross": 0.75},
            "3D ResNet": {"same": 0.89, "cross": 0.7923},
            "Hybrid Model (5F)": {"same": 0.96, "cross": 0.72}
        }
    
        method_difficulty = {
            "FaceSwap": 0.9921,
            "Deepfakes": 0.9937,
            "Face2Face": 0.9877,
            "NeuralTextures": 0.9785
        }
    
        quality_impact = {
            "High Quality": 1.0,
            "Compressed": 0.85,
            "Low Resolution": 0.70
        }
    
        eval_type = "same" if "Same-Dataset" in evaluation_type else "cross"
        base_performance = model_performances[selected_model][eval_type]
        method_factor = method_difficulty[deepfake_method] / max(method_difficulty.values())
        quality_factor = quality_impact[video_quality]
    
        predicted_confidence = base_performance * method_factor * quality_factor
        predicted_confidence = min(predicted_confidence, 0.99)  # Cap at 99%
    
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Predicted Confidence</h3>
                <h2>{predicted_confidence:.1%}</h2>
                <p>Detection Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
    
        with col2:
            risk_level = "High" if predicted_confidence > 0.85 else "Medium" if predicted_confidence > 0.70 else "Low"
            risk_color = "#4CAF50" if risk_level == "High" else "#FF9800" if risk_level == "Medium" else "#F44336"
        
            st.markdown(f"""
            <div class="metric-card" style="background: {risk_color};">
                <h3>⚡ Reliability</h3>
                <h2>{risk_level}</h2>
                <p>Detection Confidence</p>
            </div>
            """, unsafe_allow_html=True)
    
        with col3:
            processing_time = np.random.uniform(1.2, 4.8) if "3D" in selected_model else np.random.uniform(0.3, 1.5)
        
            st.markdown(f"""
            <div class="metric-card">
                <h3>⏱️ Processing Time</h3>
                <h2>{processing_time:.1f}s</h2>
                <p>Per Video</p>
            </div>
            """, unsafe_allow_html=True)
    
    
        # Simulate frame-by-frame detection
        frames = np.arange(1, 31)
        base_confidence = predicted_confidence
        noise = np.random.normal(0, 0.05, 30)
        confidence_over_time = np.clip(base_confidence + noise, 0.3, 0.99)
    
        fig_confidence = go.Figure()
        fig_confidence.add_trace(go.Scatter(
            x=frames,
            y=confidence_over_time,
            mode='lines+markers',
            name='Detection Confidence',
            line=dict(color='#4CAF50', width=3),
            marker=dict(size=6)
        ))
    
        fig_confidence.add_hline(
            y=0.5, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Decision Threshold"
        )
    
        fig_confidence.update_layout(
            title="Frame-by-Frame Detection Confidence",
            xaxis_title="Frame Number",
            yaxis_title="Detection Confidence",
            height=400,
            yaxis=dict(range=[0, 1])
        )
    
        st.plotly_chart(fig_confidence, use_container_width=True)
    
    # Analysis explanation
    st.markdown("### 📊 Analysis Explanation")
    
    explanation_col1, explanation_col2 = st.columns(2)
    
    with explanation_col1:
        st.markdown(f"""
        **Model Characteristics:**
        - **Architecture:** {selected_model}
        - **Evaluation:** {evaluation_type}
        - **Strengths:** {'Temporal analysis' if '3D' in selected_model else 'Spatial feature detection'}
        - **Best Use Case:** {'Video sequences' if '3D' in selected_model else 'Single frame analysis'}
        """)
    
    with explanation_col2:
        st.markdown(f"""
        **Detection Factors:**
        - **Method Difficulty:** {deepfake_method} (AUC: {method_difficulty[deepfake_method]:.4f})
        - **Quality Impact:** {video_quality} ({quality_impact[video_quality]:.0%} performance)
        - **Domain:** {'Same domain' if eval_type == 'same' else 'Cross domain'} evaluation
        - **Confidence:** {predicted_confidence:.1%} overall detection rate
        """)

elif st.session_state.page == "demo":
    st.markdown("## 🎥 Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📹 Project Demo Video")
        st.markdown("""
        <div class="highlight-box">
        <p>Watch our comprehensive 2-minute project demonstration:</p>
        <ul>
            <li>🎯 Problem introduction and motivation</li>
            <li>👥 Team member introductions</li>
            <li>🔬 Solution methodology overview</li>
            <li>📊 Results and model performance</li>
            <li>🚀 Live system demonstration</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
    <a href="https://drive.google.com/file/d/1MoqXYSs4OjhdZgXk2_03t_qYQx0FxBEo/view?usp=sharing" target="_blank">
        <button style="background-color:#002855; color:white; padding:0.5em 1em; border:none; border-radius:8px; font-weight:bold;">
            📺 Here is our video!
        </button>
    </a>
""", unsafe_allow_html=True)
        
    
    with col2:
        st.markdown("### 🖼️ Project Poster")
        st.markdown("""
        <div class="highlight-box">
        <p>Our research poster includes:</p>
        <ul>
            <li>📋 Clear problem definition</li>
            <li>🔬 Detailed methodology</li>
            <li>📊 Comprehensive results</li>
            <li>📈 Model comparisons</li>
            <li>🎯 Key findings and conclusions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
    <a href="https://github.com/senaylc/DeepFakeWeb/blob/master/final-poster.pdf" target="_blank">
        <button style="background-color:#002855; color:white; padding:0.5em 1em; border:none; border-radius:8px; font-weight:bold;">
            📄 Here is our poster!
        </button>
    </a>
""", unsafe_allow_html=True)
        
    
    st.markdown("### 🔗 Interactive Application")
    st.markdown("""
    <div class="highlight-box">
    <h4>🚀 Try Our DeepFake Detection System</h4>
    <p>Experience detecting deepfakes in action with our interactive web application:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯Features:**
        - Upload video analysis
        - Fast detection
        - Confidence scoring
        """)

    
    with col2:
        st.markdown("""
        **📊 Outputs:**
        - Detection probability
        - Confidence intervals
        - Visual explanations
        - Performance metrics
        """)
    
    # Link to your existing app
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <a href="https://deepfakeproject.streamlit.app/" target="_blank" 
           style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 1rem 2rem; text-decoration: none; 
                  border-radius: 8px; font-weight: bold; font-size: 1.1em;">
            🚀 Launch DeepFake Detection App
        </a>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "resources":
    st.markdown("## 📄 Project Resources")
    

    st.markdown("### 💻 Code Repository")
    st.markdown("""
    <div class="highlight-box">
    <h4>🔗 GitHub Repository</h4>
    <p>Access our complete source code, trained models, and documentation:</p>
    <ul>
        <li>📂 Complete source code</li>
        <li>🧠 Trained model weights</li>
        <li>📊 Dataset preprocessing scripts</li>
        <li>📈 Evaluation notebooks</li>
        <li>📋 Documentation and setup guides</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
        
    
    st.markdown("[🔗 View on GitHub](https://github.com/b2220765049/DeepFakeProject)")
    
    
    st.markdown("### 📚 Technical Documentation")
    
    doc_tabs = st.tabs(["📊 Dataset Information", "🧠 Model Architectures", "📈 Evaluation Metrics", "🔬 Research Papers"])
    
    with doc_tabs[0]:
        st.markdown("""
        <div class="methodology-section">
        <h4>🎬 FaceForensics++ Dataset</h4>
        <ul>
            <li><strong>Size:</strong> 1,000 original + 4,000 manipulated videos</li>
            <li><strong>Methods:</strong> FaceSwap, Deepfakes, Face2Face, NeuralTextures</li>
            <li><strong>Quality:</strong> Raw, HQ, LQ compression levels</li>
            <li><strong>Usage:</strong> Primary training and same-dataset evaluation</li>
        </ul>
        </div>
        
        <div class="methodology-section">
        <h4>🌟 Celeb-DF Dataset</h4>
        <ul>
            <li><strong>Size:</strong> 590 original + 5,639 DeepFake videos</li>
            <li><strong>Quality:</strong> High-quality, celebrity-focused</li>
            <li><strong>Diversity:</strong> Multiple ethnicities and ages</li>
            <li><strong>Usage:</strong> Cross-dataset generalization testing</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with doc_tabs[1]:
        arch_col1, arch_col2 = st.columns(2)
        
        with arch_col1:
            st.markdown("""
            **2D CNN Architectures:**
            - **ResNet50:** Deep residual learning, skip connections
            - **EfficientNet:** Compound scaling, efficient architecture
            - **XceptionNet:** Depthwise separable convolutions
            
            **3D CNN Architectures:**
            - **r3d_18:** 3D ResNet for temporal modeling
            - **Slow ResNet:** SlowFast network adaptation
            - **Custom 3D ResNet:** Modified for DeepFake detection
            """)
        
        with arch_col2:
            st.markdown("""
            **Vision Transformers:**
            - **ViT-16:** Vision Transformer with 16×16 patches
            - **Swin-S:** Hierarchical Vision Transformer
            - **ConvNeXt-S:** CNN architecture with Transformer elements
            
            **Hybrid Models:**
            - **CNN + Transformer:** Combined spatial-temporal analysis
            - **Ensemble Methods:** Multiple model voting systems
            """)
    
    with doc_tabs[2]:
        st.markdown("""
        <div class="highlight-box">
        <h4>📊 Primary Evaluation Metrics</h4>
        <ul>
            <li><strong>ROC AUC Score:</strong> Area under the ROC curve - primary metric</li>
            <li><strong>Accuracy:</strong> Correct predictions / Total predictions</li>
            <li><strong>Precision:</strong> True Positives / (True Positives + False Positives)</li>
            <li><strong>Recall:</strong> True Positives / (True Positives + False Negatives)</li>
            <li><strong>F1-Score:</strong> Harmonic mean of Precision and Recall</li>
        </ul>
        </div>
        
        <div class="cross-dataset-box">
        <h4>🎯 Evaluation Strategies</h4>
        <ul>
            <li><strong>Same-Dataset:</strong> Train and test on FaceForensics++</li>
            <li><strong>Cross-Dataset:</strong> Train on FaceForensics++, test on Celeb-DF</li>
            <li><strong>Per-Method Analysis:</strong> Individual DeepFake method performance</li>
            <li><strong>Perturbation Testing:</strong> Robustness under quality degradation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with doc_tabs[3]:
        st.markdown("""
### 📖 Key Research References

**DeepFake Detection Surveys:**

1. **Agarwal, S., Farid, H., Gu, Y., He, M., Nagano, K., & Li, H.** (2020). Detecting deepfakes: A survey. *ACM Computing Surveys*, 54(1), 1-41.
2. **Verdoliva, L.** (2020). Media forensics and deepfakes: An overview. *IEEE Journal of Selected Topics in Signal Processing*, 14(5), 910-932.

**Foundational Dataset Papers:**

3. **Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M.** (2019). FaceForensics++: Learning to detect manipulated facial images. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 1-11.
4. **Sabir, E., Cheng, J., Jaiswal, A., AbdAlmageed, W., Masi, I., & Natarajan, P.** (2019). Recurrent convolutional strategies for face manipulation detection in videos. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 18-26.

**Architecture Papers:**

5. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778.
6. **Tan, M., & Le, Q. V.** (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the International Conference on Machine Learning (ICML)*, 6105-6114.
7. **Chollet, F.** (2017). Xception: Deep learning with depthwise separable convolutions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1251-1258.
8. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al.** (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *International Conference on Learning Representations (ICLR)*.
9. **Liu, Z., Lin, Y., Cao, Y., et al.** (2021). Swin transformer: Hierarchical vision transformer using shifted windows. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 10012-10022.
10. **Liu, Z., Mao, H., Wu, C. Y., et al.** (2022). A convnet for the 2020s. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 11976-11986.

**Specialized DeepFake Detection:**

11. **Wodajo, D., & Atnafu, S.** (2021). Deepfake video detection using convolutional vision transformer. *arXiv preprint arXiv:2102.11126*.
12. **Carreira, J., & Zisserman, A.** (2017). Quo vadis, action recognition? A new model and the kinetics dataset. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 6299-6308.
""")
    
    
    
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🎓 <strong>DeepFake Detection Research Project</strong></p>
    <p>Hacettepe University | Computer/AI Engineering | BBM480 Design Project</p>
    <p>Fall 2024 - Spring 2025 | Sena Yalçın • İlayda Zeynep Karakaş • Emre Büyükyılmaz | Guided by: Nazlı İkizler Cinbiş</p>
</div>
""", unsafe_allow_html=True)                