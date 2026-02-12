
import streamlit as st

def set_custom_style():
    """Inject custom CSS for a full ChatGPT-inspired Dark Theme that sticks."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global Body Styling - Forcing Dark Theme everywhere */
        html, body, .stApp, 
        [data-testid="stAppViewContainer"], 
        [data-testid="stHeader"], 
        [data-testid="stMainViewContainer"],
        .main, .block-container {
            background-color: #343541 !important;
            color: #ECECF1 !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Sidebar - Deep Dark */
        section[data-testid="stSidebar"], 
        [data-testid="stSidebarNav"],
        section[data-testid="stSidebar"] > div {
            background-color: #202123 !important;
            border-right: 1px solid #4d4d4f !important;
        }
        
        /* Fix for potential white backgrounds in sidebar components */
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #ffffff !important;
            font-weight: 500 !important;
        }

        /* Sidebar Radio Buttons and Selectors */
        section[data-testid="stSidebar"] [data-baseweb="radio"] div,
        section[data-testid="stSidebar"] div[data-baseweb="select"] {
            color: #ffffff !important;
        }

        /* Sidebar Status Success Boxes */
        section[data-testid="stSidebar"] .stSuccess {
            background-color: #2e3034 !important;
            color: #10a37f !important;
            border: 1px solid #10a37f !important;
        }
        
        /* Main Area Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        /* Text elements */
        p, span, label, li, small {
            color: #ECECF1 !important;
        }

        /* Primary Buttons (ChatGPT Green) */
        .stButton > button {
            background-color: #10a37f !important;
            color: white !important;
            border-radius: 6px !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        .stButton > button:hover {
            background-color: #1a7f64 !important;
            color: white !important;
            transform: scale(1.01) !important;
        }

        /* Metrics Styling */
        [data-testid="stMetric"] {
            background-color: #444654 !important;
            padding: 15px !important;
            border-radius: 10px !important;
            border: 1px solid #565869 !important;
        }
        
        [data-testid="stMetricValue"] {
            color: #10a37f !important;
            font-weight: 700 !important;
        }

        /* Expanders */
        .stExpander, [data-testid="stExpander"] {
            background-color: #444654 !important;
            border: 1px solid #565869 !important;
            border-radius: 8px !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px !important;
            background-color: #202123 !important;
            padding: 5px !important;
            border-radius: 8px !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #343541 !important;
            color: #10a37f !important;
            font-weight: 600 !important;
        }

        /* Inputs and Selects */
        [data-baseweb="select"] > div,
        [data-baseweb="input"] input,
        textarea {
            background-color: #444654 !important;
            color: #ffffff !important;
            border-color: #565869 !important;
        }

        /* Tooltips and Menus */
        [data-testid="stTooltipContent"] {
            background-color: #202123 !important;
            color: #ffffff !important;
        }

        /* Dataframes */
        .stDataFrame, [data-testid="stDataFrame"] {
            background-color: #444654 !important;
            border-radius: 8px !important;
        }

        /* Hide Streamlit components that often cause white flashes */
        header[data-testid="stHeader"] {
            background: rgba(52, 53, 65, 0.8) !important;
        }

        </style>
    """, unsafe_allow_html=True)

def create_header():
    """Create a clean header for the app."""
    st.markdown("<h1>🔮 Explainable ML Playground</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; color: #C5C5D2; margin-bottom: 2rem; font-size: 1.1rem;'>
        Build, analyze, and deploy Machine Learning models with zero code.
        </div>
        """, 
        unsafe_allow_html=True
    )

def create_step_header(step_num, title, description):
    """Create a consistent header for each step."""
    st.markdown(f"### Step {step_num}: {title}")
    if description:
        st.info(description)
