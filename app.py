import os
import shutil
import pathlib
import streamlit as st
from easy_rag.RAG import get_answer  # Adjusted import to align with renamed function

st.markdown(
    """
    <style>
    /* Set the background color for the entire page */
    .stApp {
        background-color: #145835; 
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align: center; color: beige;'>Enkel RAG</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: beige;'>Chatta med ditt dokument eller Wikipedia </h3>", unsafe_allow_html=True)

# Initialize 'data_source' in session state if not already set
if 'data_source' not in st.session_state:
    st.session_state.data_source = ""

# Directory setup for uploaded documents
upload_folder = pathlib.Path().absolute() / "docs/"
if upload_folder.exists():
    shutil.rmtree(upload_folder)
upload_folder.mkdir(parents=True, exist_ok=True)

# Selection box in a colored container for enhanced appearance
with st.container():
    st.markdown(
        """
        <div style="background-color: black; padding: 15px; border-radius: 10px;">
            <h4 style='text-align: center; color: beige;'>Välj mellan document och Wikipedia for query processing:</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    placeholder = st.empty()
    source_selection = placeholder.selectbox(
        key="Källa val",
        label="Data källa:",
        options=("document", "Wikipedia")
    )

# Set session state based on selection when "Select" button is clicked
if st.button("Confirm Selection"):
    selected_option = source_selection
    st.session_state.data_source = selected_option
    st.markdown(f"<p style='color: white;'>Selected option: {st.session_state.data_source}</p>", unsafe_allow_html=True)

# Function for Wikipedia query input
def wiki_query_input() -> str:
    st.markdown("<h5>Wikipedia Search Query</h5>", unsafe_allow_html=True)
    query = st.text_input(
        label="Enter your Wikipedia search query",
        max_chars=256
    )
    if st.button("Submit Query"):
        return query

# Function to handle document upload and query for document
def upload_and_query() -> str:
    st.markdown("<h5>document Query and Document Upload</h5>", unsafe_allow_html=True)
    with st.form(key="upload_form", clear_on_submit=False):
        # Document upload field
        doc_upload = st.file_uploader(
            label="Upload your PDF document",
            accept_multiple_files=False,
            type=['pdf']
        )
        # Text input for document query
        paper_query = st.text_input(
            label="Enter your search query",
            max_chars=256
        )
        submit_button = st.form_submit_button("Upload and Process")

    # Save the uploaded document and return the query
    if submit_button and doc_upload is not None:
        with open(os.path.join(upload_folder, doc_upload.name), 'wb') as file:
            file.write(doc_upload.getbuffer())
        return paper_query

# Main function to manage query processing based on selection
def process_selection(data_source):
    if data_source == "Wikipedia":
        query = wiki_query_input()
        if query:
            with st.spinner("Processing your Wikipedia query..."):
                # Fetch answer for Wikipedia
                answer = get_answer(data_source, query)
                st.success("Processing complete!")
                # Display answer or fallback message if 'answer' key is missing
                st.markdown(
                    f"<div style='background-color: black; padding: 10px; border-radius: 5px;'>"
                    f"<h5>Answer:</h5> {answer.get('answer', 'No answer available')}"
                    f"</div>", 
                    unsafe_allow_html=True
                )

    elif data_source == "document":
        query = upload_and_query()
        if query:
            with st.spinner("Processing your research document query..."):
                # Fetch answer for document
                answer = get_answer(data_source, query)
                st.success("Processing complete!")
                st.markdown(
                    f"<div style='background-color: black; padding: 10px; border-radius: 5px;'>"
                    f"<h5>Answer:</h5> {answer.get('answer', 'No answer available')}"
                    f"</div>", 
                    unsafe_allow_html=True
                )

# Execute the main function based on session state
if __name__ == "__main__":
    process_selection(st.session_state.data_source)
