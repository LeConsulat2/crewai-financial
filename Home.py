import streamlit as st

st.title("CrewAI!")

# Display welcome message
st.markdown(
    """
    # Kia Ora!
    Welcome to CrewAI Page!

    👈Please access the CREWAI from the sidebar!:
    """
)

# Navigation options
st.markdown(
    """
    - [Online Form Crew](pages/online-form-crew.py)
    - [Securedocs Crew](pages/securedoc-crew.py)
    """
)

# Optionally, add buttons for navigation
# if st.button("Go to Online Form Crew"):
#     st.experimental_set_query_params(page="Online Form Crew")
#     st.markdown("[Click here to go to Online Form Crew](pages/online-form-crew.py)")

# if st.button("Go to Securedocs Crew"):
#     st.experimental_set_query_params(page="Securedocs Crew")
#     st.markdown("[Click here to go to Securedocs Crew](pages/securedoc-crew.py)")
