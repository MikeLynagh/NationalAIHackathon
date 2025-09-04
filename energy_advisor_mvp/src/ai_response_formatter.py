import streamlit as st

def format_ai_response(ai_response: str) -> None:
    """
    Format AI response with better styling - improved readability version
    """
    if not ai_response or len(ai_response.strip()) < 50:
        return
    
    # Clean up the response
    cleaned_response = ai_response.strip()
    
    # Create a styled container with darker, more readable text
    with st.container():
        st.markdown("---")
        st.markdown("### 🤖 AI-Powered Energy Analysis")
        st.markdown("*Powered by DeepSeek AI*")
        
        # Display the response with improved formatting and darker text
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="
                white-space: pre-wrap; 
                line-height: 1.6;
                color: #2c3e50;
                font-size: 14px;
                font-weight: 400;
            ">
                {cleaned_response}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a call-to-action with darker text
        st.markdown("""
        <div style="
            background: #e8f4fd;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            margin: 1rem 0;
        ">
            <p style="color: #2c3e50; margin: 0; font-weight: 500;">
                💡 <strong>Next Steps:</strong> Review the recommendations above and consider implementing the suggested changes to optimize your energy usage.
            </p>
        </div>
        """, unsafe_allow_html=True)
