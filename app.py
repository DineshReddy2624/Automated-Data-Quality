import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Setting up the page - learned this from online tutorials
st.set_page_config(page_title="My Data Cleaning Tool", page_icon="📊", layout="wide")

# Custom CSS styling - took me forever to get the colors right!
st.markdown("""
    <style>
        /* Main font family - I prefer Segoe UI */
        html, body, [class*="st-"] {
            font-family: 'Segoe UI', 'Montserrat', Arial, sans-serif !important;
        }
        
        /* Selectbox styling - this was tricky to figure out */
        .stSelectbox > div > div {
            background-color: #f7fbff !important;
            color: #173055 !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
        }
        .stSelectbox span {
            color: #173055 !important;
            font-weight: 600 !important;
        }
        
        /* Dropdown styling */
        .stSelectbox [data-baseweb="popover"] {
            background-color: #f7fbff !important;
            color: #173055 !important;
            border-radius: 8px !important;
        }
        .stSelectbox [data-baseweb="option"] {
            background-color: #e3ecfc !important;
            color: #173055 !important;
            font-weight: 600 !important;
        }
        .stSelectbox [data-baseweb="option"]:hover {
            background-color: #b1d4ff !important;
            color: #112244 !important;
        }
        
        /* Text color - white looks good on my background */
        h1, h2, h3, h4, h5, h6, .stMarkdown, label, p, span {
            color: white !important;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: #f7fbff !important;
            color: #0e2544 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #0e2544 !important;
        }
        
        /* Button styling - blue theme */
        .stButton > button {
            background: #1e90ff !important;
            color: #fff !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }
        .stButton > button:hover {
            background: #004aad !important;
        }
        
        /* DataFrame styling */
        .stDataFrame th, .stDataFrame td {
            background-color: rgba(255,255,255,0.92) !important;
            color: #112244 !important;
            font-weight: 600 !important;
        }
        
        /* File uploader text - this was a pain to style properly */
        section[data-testid="stFileUploadDropzone"] p,
        section[data-testid="stFileUploadDropzone"] span,
        section[data-testid="stFileUploadDropzone"] div,
        .stFileUploader div[role="button"] span,
        .stFileUploader div[role="button"] p {
            color: white !important;
            font-weight: 600 !important;
        }
        
        section.css-1oyhl2h * {
            color: white !important;
        } 
    </style>
""", unsafe_allow_html=True)

# Function to set background image - copied this from Stack Overflow and modified
def add_bg_image(img_path):
    try:
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        background_css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(background_css, unsafe_allow_html=True)
    except:
        # If image doesn't exist, just continue without background
        pass

# My custom background image
add_bg_image("D:\E  Drive\Automatic data cleaning(Final Year Project)\IMages\Background.png")

# Session state initialization - need this for maintaining data across reruns
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'cleaning_steps' not in st.session_state:
    st.session_state.cleaning_steps = []
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None

# Sidebar for file upload and controls
with st.sidebar:
    st.title("File Upload & Options")
    
    # File uploader
    csv_file = st.file_uploader("Choose your CSV file", type=["csv"])
    
    # Sample data download - useful for testing
    if st.button("Get Sample Data"):
        sample_data = "Name,Age,City,Salary\nJohn,25,Mumbai,50000\nJane,,Delhi,60000\nBob,30,,55000\nAlice,28,Chennai,65000"
        st.download_button("Download Sample", sample_data, "test_data.csv", "text/csv")
    
    # Reset button
    clear_all = st.button("Reset Values")

# Main application header
st.title("Data Cleaning Tool")

# Main logic starts here
if csv_file and not clear_all:
    # Load the uploaded file
    if st.session_state.raw_data is None or csv_file.name != getattr(st.session_state, 'current_file', None):
        try:
            original_df = pd.read_csv(csv_file)
            
            # Handle different ways people represent missing data
            missing_values = ['None', 'none', 'NONE', 'null', 'NULL', 'nan', 'NaN', 'N/A', 'n/a', '', ' ', 'na', 'NA']
            original_df = original_df.replace(missing_values, np.nan)
            
            st.session_state.raw_data = original_df.copy()
            st.session_state.cleaned_data = original_df.copy()
            st.session_state.cleaning_steps = []
            st.session_state.current_file = csv_file.name
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()
    
    # Get the data
    original_df = st.session_state.raw_data
    current_df = st.session_state.cleaned_data.copy()

    # Quick stats section
    st.subheader("Dataset Overview")
    
    # Create metrics columns
    metric1, metric2, metric3, metric4 = st.columns(4)
    
    total_rows = len(current_df)
    total_cols = len(current_df.columns)
    missing_count = current_df.isnull().sum().sum()
    missing_percent = round((missing_count / current_df.size) * 100, 1)
    duplicate_rows = current_df.duplicated().sum()
    
    metric1.metric("Total Rows", total_rows)
    metric2.metric("Total Columns", total_cols)
    metric3.metric("Missing Values %", f"{missing_percent}%")
    metric4.metric("Duplicate Rows", duplicate_rows)

    # Show what cleaning steps have been applied
    if len(st.session_state.cleaning_steps) > 0:
        st.info(f"Applied operations: {', '.join(st.session_state.cleaning_steps)}")

    # Data preview section
    with st.expander("Take a Look at Your Data"):
        st.dataframe(current_df.head(15), use_container_width=True)

    # Missing data analysis
    with st.expander("Missing Data Details"):
        missing_data = current_df.isnull().sum()
        cols_with_missing = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(cols_with_missing) > 0:
            st.write("**Found missing values in these columns:**")
            for column, count in cols_with_missing.items():
                percent = (count / len(current_df)) * 100
                st.write(f"• **{column}**: {count} missing values ({percent:.1f}%)")
        else:
            st.success("Great news! No missing values detected.")

    # Profiling report section
    st.subheader("Detailed Analysis Report")
    
    if st.button("Create Detailed Report"):
        with st.spinner("Creating your analysis report... this might take a moment"):
            try:
                report = ProfileReport(current_df, title="Dataset Analysis", explorative=True)
                st_profile_report(report)
            except Exception as e:
                st.error(f"Couldn't generate report: {str(e)}")

    # Data cleaning options in sidebar
    with st.sidebar:
        st.markdown("### Clean Your Data")
        
        # Find columns with missing values
        cols_with_nulls = current_df.columns[current_df.isnull().any()].tolist()
        
        if len(cols_with_nulls) > 0:
            st.warning(f" Found {len(cols_with_nulls)} columns with missing data")
            
            # Show details of missing data
            with st.expander("Which columns have missing data?"):
                for col in cols_with_nulls:
                    null_count = current_df[col].isnull().sum()
                    st.write(f"• {col}: {null_count} missing")
            
            # Cleaning strategy selection
            strategy = st.selectbox("How do you want to handle missing values?",
                                  ["Remove rows with any missing values", 
                                   "Fill numbers with average (mean)", 
                                   "Fill numbers with middle value (median)",
                                   "Fill everything with most common value",
                                   "Fill everything with zeros",
                                   "Remove columns that are mostly empty (>50%)"])

            # Apply the selected strategy
            if st.button("Clean the Data!"):
                df_to_clean = current_df.copy()
                
                if strategy == "Remove rows with any missing values":
                    df_to_clean = df_to_clean.dropna()
                    operation_name = "Dropped missing rows"
                    
                elif strategy == "Fill numbers with average (mean)":
                    # Handle numeric columns
                    numeric_columns = df_to_clean.select_dtypes(include=[np.number]).columns
                    df_to_clean[numeric_columns] = df_to_clean[numeric_columns].fillna(df_to_clean[numeric_columns].mean())
                    # Handle text columns
                    text_columns = df_to_clean.select_dtypes(include=['object']).columns
                    df_to_clean[text_columns] = df_to_clean[text_columns].fillna('0')
                    operation_name = "Filled with mean/0"
                    
                elif strategy == "Fill numbers with middle value (median)":
                    numeric_columns = df_to_clean.select_dtypes(include=[np.number]).columns
                    df_to_clean[numeric_columns] = df_to_clean[numeric_columns].fillna(df_to_clean[numeric_columns].median())
                    text_columns = df_to_clean.select_dtypes(include=['object']).columns
                    df_to_clean[text_columns] = df_to_clean[text_columns].fillna('0')
                    operation_name = "Filled with median/0"
                    
                elif strategy == "Fill everything with most common value":
                    for column in df_to_clean.columns:
                        most_common = df_to_clean[column].mode()
                        if len(most_common) > 0:
                            df_to_clean[column] = df_to_clean[column].fillna(most_common[0])
                        else:
                            df_to_clean[column] = df_to_clean[column].fillna('0')
                    operation_name = "Filled with mode"
                    
                elif strategy == "Fill everything with zeros":
                    df_to_clean = df_to_clean.fillna(0)
                    operation_name = "Filled with zeros"
                    
                elif strategy == "Remove columns that are mostly empty (>50%)":
                    threshold_value = len(df_to_clean) * 0.5
                    df_to_clean = df_to_clean.dropna(axis=1, thresh=threshold_value)
                    df_to_clean = df_to_clean.fillna('0')
                    operation_name = "Removed empty columns"
                
                # Update session state
                st.session_state.cleaned_data = df_to_clean
                if operation_name not in st.session_state.cleaning_steps:
                    st.session_state.cleaning_steps.append(operation_name)
                
                st.success(f"Done! Applied: {operation_name}")
                st.experimental_rerun()
        else:
            st.success("No missing values found - your data looks clean!")

    # Handle duplicate data
    with st.sidebar:
        st.markdown("### Handle Duplicates")
        
        duplicate_count = current_df.duplicated().sum()
        if duplicate_count > 0:
            st.warning(f"Found {duplicate_count} duplicate rows")
            
            if st.button("Remove Duplicates"):
                st.session_state.cleaned_data = current_df.drop_duplicates()
                if "Removed duplicates" not in st.session_state.cleaning_steps:
                    st.session_state.cleaning_steps.append("Removed duplicates")
                st.success(f"Removed {duplicate_count} duplicate rows!")
                st.experimental_rerun()
        else:
            st.success(" No duplicates found!")

    # Simple visualizations
    with st.expander(" Quick Charts"):
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Pick a column to visualize", numeric_cols)
            
            if selected_col:
                # Create histogram
                plt.figure(figsize=(10, 6))
                sns.histplot(current_df[selected_col].dropna(), bins=25, kde=True)
                plt.title(f"Distribution of {selected_col}")
                plt.xlabel(selected_col)
                plt.ylabel("Frequency")
                st.pyplot(plt)
        else:
            st.info("No numeric columns found for visualization")

    # Download section
    st.subheader(" Download Your Cleaned Data")
    
    # Show before/after comparison
    original_row_count = len(original_df)
    cleaned_row_count = len(current_df)
    original_missing = int(original_df.isnull().sum().sum())
    cleaned_missing = int(current_df.isnull().sum().sum())
    
    # Display comparison metrics
    comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
    comparison_col1.metric("Original Rows", original_row_count)
    comparison_col2.metric("Cleaned Rows", cleaned_row_count, delta=cleaned_row_count - original_row_count)
    comparison_col3.metric("Missing Values", cleaned_missing, delta=cleaned_missing - original_missing)
    
    # Show cleaning effectiveness message
    if cleaned_missing == 0:
        st.success("🎉 Perfect! Your dataset is now completely clean with no missing values!")
    else:
        remaining_missing_percent = (cleaned_missing / (cleaned_row_count * len(current_df.columns))) * 100
        if remaining_missing_percent < 3:
            st.success(f" Almost there! Your dataset is {100-remaining_missing_percent:.1f}% clean now.")
        else:
            st.info(f" Your dataset still has {remaining_missing_percent:.1f}% missing values. You might want to try other cleaning methods.")
    
    # Create download button
    cleaned_csv_data = current_df.to_csv(index=False)
    st.download_button(
        label="Download Cleaned Data",
        data=cleaned_csv_data.encode('utf-8'),
        file_name=f"cleaned_{csv_file.name}",
        mime="text/csv",
        help="Click to download your cleaned dataset as a CSV file"
    )

elif not csv_file and not clear_all:
    # About the App section
    st.markdown("""
    ##About This App

    The **Data Cleaning Tool (Data Detox)** is designed to help users quickly **analyze, clean, and prepare datasets** for analysis or machine learning.  
    It provides an easy-to-use interface where you can:

    - Upload your dataset (CSV format)  
    - Explore data quality through automatic profiling reports  
    - Handle missing values with different strategies (drop, mean, median, mode, zero-fill, remove columns)  
    - Detect and remove duplicate records  
    - Identify anomalies and outliers using visualization techniques  
    - Generate detailed profiling reports with insights like correlations, distributions, and summary statistics  
    - Download the cleaned dataset for further use  

    --

    ##  How It Works (Process Flow)

    1. **Upload** – User uploads a dataset (CSV file) through the sidebar.  
    2. **Analyze** – The app calculates basic statistics (rows, columns, missing values, duplicates) and shows a preview.  
    3. **Clean** – Choose strategies for handling missing data and duplicates from the sidebar options.  
    4. **Visualize** – Explore missing value matrices, outlier box plots, correlation heatmaps, and profiling reports.  
    5. **Download** – Once the data is cleaned, the user can download the processed dataset for further use.  

    This app reduces manual effort and ensures that your data is **clean, consistent, and ready** for any data analysis or machine learning project.
    """)

# Handle reset functionality
if clear_all:
    # Clear all session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()
