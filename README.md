DocuMind - Ask Your File 📄
===========================

Overview:
----------
DocuMind is a web application built with Streamlit, which allows users to upload documents (TXT, DOCX, or PDF formats) and ask questions based on the content of the uploaded files. It utilizes powerful natural language processing models to search for answers and display the most relevant information.

Key Features:
-------------
- Upload documents in the following formats: `.txt`, `.docx`, `.pdf`.
- Ask questions related to the document's content.
- The app breaks down the document into smaller chunks for more accurate question answering.
- Displays top-matching contexts and answers with confidence scores.
- Dark Mode option for a more comfortable reading experience.
- History of questions and answers on the sidebar for easy reference.

Requirements:
-------------
To run this project locally, you will need to install the following Python libraries:
- streamlit
- torch
- transformers
- sentence-transformers
- python-docx
- PyMuPDF (fitz)

Installation:
-------------
1. Clone or download the repository.
2. Install the required libraries by running:
   ```bash
   pip install streamlit torch transformers sentence-transformers python-docx PyMuPDF


How it Works:
Document Upload:

The user uploads a document in .txt, .docx, or .pdf format.

The app extracts the text from the uploaded document for processing.

Question Answering:

The app breaks down the document into smaller chunks of text.

For each chunk, the app uses a pre-trained question-answering model to identify the most relevant answer to the user’s query.

Context Matching:

The app uses sentence embeddings to match the user's question with the most relevant contexts from the document.

Top matching chunks are displayed, followed by their corresponding answers and confidence scores.

Dark Mode:

The app offers a toggle for dark mode, providing a more comfortable experience for users in low-light environments.

History:

The sidebar stores a history of previous questions and answers, allowing users to quickly refer back to earlier queries.

Additional Notes:
The app automatically handles large documents by splitting them into manageable chunks.

Confidence scores are displayed along with each answer to indicate the reliability of the response.
