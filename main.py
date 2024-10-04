import streamlit as st

st.title("CrewAI Hello!")

# Display welcome message
st.markdown(
    """
    # Kia Ora!
    Welcome to CrewAI Financial Assistance Page (Beta)!
    This CrewAI Financial tool will help you to assess the financial needs but to also assess your consistency in the assessment.

    👈Here are the apps you can access from the Side bar:
    """
)

# Navigation options
# st.markdown(
#     """
#     - [Online Form Crew](pages/online-form-crew.py)
#     - [Securedocs Crew](pages/securedoc-crew.py)
#     """
# )

# # Optionally, add buttons for navigation
# if st.button("Go to Online Form Crew"):
#     st.experimental_set_query_params(page="Online Form Crew")
#     st.markdown("[Click here to go to Online Form Crew](pages/online-form-crew.py)")

# if st.button("Go to Securedocs Crew"):
#     st.experimental_set_query_params(page="Securedocs Crew")
#     st.markdown("[Click here to go to Securedocs Crew](pages/securedoc-crew.py)")
