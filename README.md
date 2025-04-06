# PDF & SQL QnA Bot 🤖📄📊

This project is a Streamlit web application that acts as an intelligent Question & Answering (QnA) bot. It can answer user queries based on information extracted from uploaded PDF documents or by querying a connected SQLite database. The bot leverages Google's Gemini API for natural language understanding and generation, and Sentence Transformers for efficient text embedding and retrieval.

## Features

*   **Dual Data Sources:** Answers questions using context from uploaded PDFs or data fetched from an SQLite database.
*   **PDF Processing (RAG):**
    *   Accepts multiple PDF uploads.
    *   Extracts text from PDFs.
    *   Chunks text intelligently using a sliding window approach.
    *   Generates vector embeddings for text chunks using `all-MiniLM-L6-v2`.
    *   Implements **Retrieval-Augmented Generation (RAG)**: Finds the most relevant PDF chunks based on query similarity and provides them as context to the LLM.
*   **Natural Language to SQL:**
    *   Interprets natural language questions related to the database.
    *   Uses Google Gemini to automatically generate SQL queries.
    *   Executes generated SQL against the specified SQLite database.
    *   Displays query results directly in the chat interface (as Pandas DataFrames).
*   **Intelligent Query Routing:** Automatically detects whether a query is likely intended for the SQL database (based on keywords) or the PDF documents.
*   **Embedding Caching:** Caches PDF chunk embeddings locally (`.embedding_cache/`) using text hashes to speed up processing when the same PDFs are re-uploaded.
*   **Interactive UI:** Built with Streamlit for easy configuration, file upload, and chat interaction.
*   **Configuration:** API keys and database paths can be configured via environment variables or directly in the Streamlit sidebar.

## Technology Stack

*   **Frontend:** Streamlit
*   **LLM:** Google Gemini API (`gemini-1.5-flash`)
*   **Embedding Model:** Sentence Transformers (`all-MiniLM-L6-v2`)
*   **PDF Processing:** PyPDF
*   **Database:** SQLite3
*   **Data Handling:** Pandas, NumPy
*   **Vector Similarity:** Scikit-learn (`cosine_similarity`)
*   **Core Language:** Python 3

## How It Works

1.  **Initialization:** The `SmartChatBot` class is initialized with the Gemini API key and database path. It sets up the Gemini model, the Sentence Transformer embedding model, and prepares for database connections and PDF storage.
2.  **PDF Processing:** When PDFs are uploaded, the text is extracted, broken into overlapping chunks, and embeddings are generated. These embeddings are cached locally based on a hash of the chunk text.
3.  **User Query:** The user enters a query in the chat interface.
4.  **Query Routing:** The `smart_chat` method checks if the query contains SQL-related keywords (e.g., 'table', 'database', 'query').
5.  **SQL Path:** If SQL keywords are detected, the query is sent to the Gemini model with a prompt asking it to convert the natural language query into an SQL query. The generated SQL is executed against the database using `pandas.read_sql_query`, and the results (as a DataFrame) are returned.
6.  **PDF/RAG Path:** If no SQL keywords are found:
    *   The user query is embedded using the Sentence Transformer model.
    *   Cosine similarity is calculated between the query embedding and all cached PDF chunk embeddings.
    *   The `top_k` most similar PDF chunks are retrieved.
    *   These relevant chunks are combined with the original user query into an augmented prompt for the Gemini model.
    *   Gemini generates an answer based on the provided context and its general knowledge.
7.  **Response:** The generated answer (either SQL results or text) is displayed in the Streamlit chat interface.

## Embedding Cache

To avoid redundant computation, embeddings for PDF text chunks are cached in the `.embedding_cache` directory.
*   A hash (MD5) of each text chunk is generated.
*   The corresponding embedding (converted to a list) is saved as a JSON file named `{hash}_embedding.json`.
*   Before generating an embedding for a chunk, the system checks if a cached file exists for its hash. If found, the cached embedding is loaded; otherwise, a new embedding is generated and cached.

---
