import streamlit as st


# Set the page configuration
st.set_page_config(page_title="AUT GPT Home", page_icon="💬")


st.title("CrewAI!")

# Display markdown content
st.markdown(
    """
    # Kia Ora!    
    Welcome to CrewAI Page!

    👈Here are the apps you can access from the Side bar👈:
    """
)

# Display the links to the GPT apps only if the user is authenticated

st.markdown(
    """
    - Online Form Crew
                    
    - Securedocs Crew

    

    """
)
