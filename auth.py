import streamlit as st
from home import home_page

# Handles the authentication process
def authenticate():
    """Manages the authentication logic, including session state initialization."""
    
    # Initialize session state for authentication if it doesn't exist
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Show login form if user is not authenticated
    if not st.session_state.authenticated:
        login_form()
    else:
        show_authenticated_content()

# Displays the login form
def login_form():
    """Renders the login form and handles login validation."""
    
    st.title("Login")
    # Using a form to group login fields
    with st.form("login_form"):
        username = st.text_input("Username", key="username_input")  # Input for username
        password = st.text_input("Password", type="password", key="password_input")  # Input for password

        # When the login form is submitted
        submitted = st.form_submit_button("Login")
        if submitted:
            # Basic hardcoded authentication check (replace with a more secure method in production)
            if username == "admin" and password == "password":
                st.session_state.authenticated = True  # Set session state as authenticated
                st.success("Successfully logged in!")  # Success message
            else:
                st.error("Invalid credentials")  # Error message for incorrect login

# Displays content after authentication
def show_authenticated_content():
    """Shows the content for authenticated users, including a logout option."""
    
    st.title("Welcome to the authenticated page!")
    # Logout option placed in the sidebar
    if st.sidebar.button("Logout"):
        logout()

# Handles user logout and session reset
def logout():
    """Logs out the user by resetting the authentication state."""
    
    st.session_state.authenticated = False  # Reset authenticated status
    st.success("You have been logged out!")  # Confirmation message

# Main entry point for the app
def main():
    """The main function that orchestrates the authentication flow."""
    
    authenticate()

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
