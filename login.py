import streamlit as st


# Reading credentials from secrets.toml
credentials = st.secrets["credentials"]


def login_page(navigate_to):
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if username == credentials["username"] and password == credentials["password"]:
            st.session_state.authenticated = True
            st.success("Login successful!")
            navigate_to("home")  # Navigate to home page upon successful login
        else:
            st.error("Invalid username or password")
