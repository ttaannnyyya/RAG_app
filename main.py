
import streamlit as st
from langchain_helper import get_qa_chain , send_mail

st.title("NullClass’s Smart Assistant 🤖")

# Initialize chat history list in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history as conversation bubbles
for chat in st.session_state.chat_history:
    if chat["role"] == "human":
        st.markdown(f"**You:** {chat['content']}")
    else:
        st.markdown(f"**Assistant:** {chat['content']}")

# Input box for user question
question = st.text_input("Question:", key="question_input")

if question:
    chain = get_qa_chain()
    response = chain(question)  # response is answer string now

    # Append to chat history
    st.session_state.chat_history.append({"role": "human", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display the latest answer immediately
    st.header("Answer")
    st.write(response)
# --- Exit / feedback flow ---
# initialize session state flags
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "show_email_input" not in st.session_state:
    st.session_state.show_email_input = False

# When user clicks Exit Chat, set flag to show feedback panel
if st.button("Exit Chat", key="exit_chat"):
    st.session_state.show_feedback = True
    st.session_state.show_email_input = False  # reset email stage

# Render feedback panel if requested
if st.session_state.show_feedback:
    st.subheader("Please let us know:")
    resolved = st.radio("Was your issue resolved?", ("Yes", "No"), key="resolved_radio")

    # Submit resolution choice
    if st.button("Submit", key="submit_resolution"):
        if resolved == "Yes":
            st.success("Thank you for trusting us!")   # only thank you, no feedback request
            st.session_state.show_feedback = False   # hide panel after submit
        else:
            # show email input on next render
            st.session_state.show_email_input = True

    # Email input stage (shown after unresolved + submit)
    if st.session_state.show_email_input:
        email = st.text_input("We’re sorry! Please share your email so we can follow up:", key="email_input")
        if st.button("Submit Email", key="submit_email"):
            if email.strip():
                st.success("Thank you! Our team will contact you soon. 📩")
                send_mail(email.strip())   # <-- CALL the function here (no definition included)
                # reset flags to hide the panel
                st.session_state.show_email_input = False
                st.session_state.show_feedback = False
            else:
                st.error("Please enter a valid email.")

