import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import os

# Function to compute the DH transformation matrix
def dh_transform_matrix(theta, d, a, alpha):
    T = np.array([[np.cos(theta), -np.sin(theta), 0, a * np.cos(theta)],
                  [np.sin(theta), np.cos(theta), 0, a * np.sin(theta)],
                  [0, 0, 1, d],
                  [0, 0, 0, 1]])
    return T

# Function to compute forward kinematics
def forward_kinematics(theta1, theta2, a1, a2):
    x1 = a1 * np.cos(theta1)
    y1 = a1 * np.sin(theta1)
    x2 = x1 + a2 * np.cos(theta1 + theta2)
    y2 = y1 + a2 * np.sin(theta1 + theta2)
    return x2, y2

# Function to compute inverse kinematics
def inverse_kinematics(x, y, a1, a2):
    r = np.sqrt(x**2 + y**2)
    if r > (a1 + a2):
        raise ValueError("Target is out of reach")
    
    cos_theta2 = (r**2 - a1**2 - a2**2) / (2 * a1 * a2)
    theta2 = np.arccos(np.clip(cos_theta2, -1.0, 1.0))
    
    k1 = a1 + a2 * np.cos(theta2)
    k2 = a2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    return theta1, theta2

# Visualization function
def visualize_robot(theta1, theta2, a1, a2, target_x=None, target_y=None):
    x1, y1 = a1 * np.cos(theta1), a1 * np.sin(theta1)
    x2, y2 = x1 + a2 * np.cos(theta1 + theta2), y1 + a2 * np.sin(theta1 + theta2)

    plt.figure(figsize=(6, 6))
    plt.plot([0, x1], [0, y1], label='Link 1', color='#1f77b4', linewidth=3)
    plt.plot([x1, x2], [y1, y2], label='Link 2', color='#2ca02c', linewidth=3)
    plt.scatter(x2, y2, color='#9467bd', s=100, label='End Effector Position')

    if target_x is not None and target_y is not None:
        plt.scatter(target_x, target_y, color='#d62728', s=100, label='Target Position')

    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("2D Planar 2-DOF Robot Arm")
    plt.legend()
    return plt

# Streamlit Interface
st.set_page_config(layout="centered", page_title="Robotic Arm Simulation")

# Custom CSS
st.markdown("""
    <style>
    .container {
        max-width: 1000px;
        margin: auto;
        padding: 0 1rem;
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .logo {
        width: 80px;
        height: 80px;
        margin-right: 1rem;
    }
    .title {
        color: #1f2937;
        font-size: 24px !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }
    .subtitle {
        color: #374151;
        font-size: 20px !important;
        margin: 0.5rem 0 !important;
    }
    .project-title {
        color: #dc2626;
        font-weight: bold;
        font-size: 22px !important;
        margin: 1rem 0 !important;
        text-align: center;
    }
    .team-container {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0;
        padding: 1rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .team-members {
        flex: 2;
    }
    .guided-by {
        flex: 1;
        text-align: right;
    }
    .input-container {
        margin-top: 2rem;
    }
    .stButton > button {
        background-color: #7da7d9;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 4px;
        border: none;
        margin: 1rem auto;
        display: block;
    }
    div[data-testid="stVerticalBlock"] > div {
        padding: 0.25rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header with Logo
col1, col2 = st.columns([1, 5])
with col1:
    logo_path = r"Logo.png"
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, width=80, use_container_width=False)
    else:
        st.warning("Logo not found")

with col2:
    st.markdown('<h1 class="title">BASAVESHWAR ENGINEERING COLLEGE, BAGALKOTE</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subtitle">DEPARTMENT OF MECHANICAL ENGINEERING</h2>', unsafe_allow_html=True)

# Project Title
st.markdown('<p style="text-align: center; font-size: 18px;">Project on</p>', unsafe_allow_html=True)
st.markdown('<h2 class="project-title">"ROBOTIC ARM SIMULATION USING PYTHON"</h2>', unsafe_allow_html=True)

# Team Members and Guide
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("**Team Members:**")
    st.markdown("""
    ➜ Arpita Kalligudd (2BA22ME406)  
    ➜ Karthik S Budni (2BA22ME414)  
    ➜ Praveen V Patil (2BA22ME433)  
    ➜ Raju S Sobarad (2BA22ME435)
    """)

with col2:
    st.markdown("**Guided By**")
    st.markdown("Dr. V. V. Kuppast")

# Input Parameters Section
st.markdown("### Input parameters:")

# Mode Selection
mode = st.selectbox("Mode", ["Forward Kinematics", "Inverse Kinematics"])

# Parameters input in a grid
col1, col2, col3, col4 = st.columns(4)

with col1:
    a1 = st.number_input("Link 1 Length", value=3.0, min_value=0.1, max_value=5.0, step=0.1)

with col2:
    a2 = st.number_input("Link 2 Length", value=2.0, min_value=0.1, max_value=5.0, step=0.1)

if mode == "Forward Kinematics":
    with col3:
        theta1_deg = st.number_input("Theta 1", value=45.0, min_value=-180.0, max_value=180.0, step=1.0)
    with col4:
        theta2_deg = st.number_input("Theta 2", value=45.0, min_value=-180.0, max_value=180.0, step=1.0)

    if st.button("SIMULATE"):
        theta1 = np.radians(theta1_deg)
        theta2 = np.radians(theta2_deg)
        x, y = forward_kinematics(theta1, theta2, a1, a2)
        
        st.write(f"End Effector Position: x = {x:.2f}, y = {y:.2f}")
        plt = visualize_robot(theta1, theta2, a1, a2)
        st.pyplot(plt)

else:  # Inverse Kinematics
    with col3:
        target_x = st.number_input("Target X", value=2.0, min_value=-5.0, max_value=5.0, step=0.1)
    with col4:
        target_y = st.number_input("Target Y", value=2.0, min_value=-5.0, max_value=5.0, step=0.1)

    if st.button("SIMULATE"):
        try:
            theta1, theta2 = inverse_kinematics(target_x, target_y, a1, a2)
            st.write(f"Joint 1 Angle (Theta1): {np.degrees(theta1):.2f} degrees")
            st.write(f"Joint 2 Angle (Theta2): {np.degrees(theta2):.2f} degrees")
            
            plt = visualize_robot(theta1, theta2, a1, a2, target_x, target_y)
            st.pyplot(plt)
        except ValueError as e:
            st.error(str(e))
