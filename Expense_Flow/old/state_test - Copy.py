import streamlit as st

# Must be first Streamlit command
st.set_page_config(page_title="State Test", page_icon="🔍")

def main():
    st.title("Session State Test")
    
    # Display current count
    st.write(f"Current count: {st.session_state.get('count', 0)}")
    
    # Button to increment count
    if st.button("Increment Count"):
        if 'count' not in st.session_state:
            st.session_state.count = 0
        st.session_state.count += 1
    
    # Display entire session state
    st.write("Full session state:", st.session_state)
    
    # Add a text input to test more state
    text = st.text_input("Enter some text", key="text_input")
    st.write("You entered:", text)

if __name__ == "__main__":
    main()