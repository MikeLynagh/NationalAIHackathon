import streamlit as st

def add_custom_css():
    """
    Add custom CSS styling for better visual appeal
    """
    st.markdown("""
    <style>
    /* AI Response Styling */
    .ai-response-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    
    .ai-response-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0 1rem 0;
    }
    
    /* Recommendation Cards */
    .recommendation-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .recommendation-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Impact Level Styling */
    .high-impact {
        border-left: 4px solid #28a745;
        background-color: #d4edda;
    }
    
    .medium-impact {
        border-left: 4px solid #17a2b8;
        background-color: #d1ecf1;
    }
    
    .low-impact {
        border-left: 4px solid #ffc107;
        background-color: #fff3cd;
    }
    
    .minimal-impact {
        border-left: 4px solid #6c757d;
        background-color: #f8f9fa;
    }
    
    /* Savings Highlight */
    .savings-highlight {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem 0;
    }
    
    /* Action Items Styling */
    .action-item {
        background: #e3f2fd;
        border-left: 3px solid #2196f3;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    /* Section Headers */
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .recommendation-card {
            margin: 0.25rem 0;
            padding: 0.75rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
