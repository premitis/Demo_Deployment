import streamlit as st
import pdfplumber
import os
from groq import Groq
from utils import model

# Function to interact with GROQ API and get response based on extracted PDF text
def get_doc_response(user_input):
    os.environ['GROQ_API_KEY'] = model

    client = Groq()
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": """Answer the question as detailed as possible from the provided context,
                    make sure to provide all the details, if the answer is not in the provided context, just say, 
                    "Answer is not available in the context", don't provide the wrong answer"""
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    output = ""
    for chunk in completion:
        output += chunk.choices[0].delta.content or ""
    return output

# Function to extract text from uploaded PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Streamlit App
def main():
    st.title("Chat With PDF")

    # Upload PDF file
    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if pdf_file is not None:
        # Extract text from the PDF
        with st.spinner("Extracting text from the PDF..."):
            extracted_text = extract_text_from_pdf(pdf_file)
            st.write("Extracted Text from PDF:")
            st.text_area("PDF Content", extracted_text, height=200)

        # User Input for Chat
        st.write("Chat with PDF:")
        user_input = st.text_input("Ask you questions:")

        if st.button("Get Response"):
            if user_input:
                with st.spinner("Generating response..."):
                    response = get_doc_response(extracted_text + "\n" + user_input)
                    st.write("AI Response:")
                    st.text_area("Response", response, height=200)
            else:
                st.warning("Please ask a question based on the PDF.")

if __name__ == "__main__":
    main()
