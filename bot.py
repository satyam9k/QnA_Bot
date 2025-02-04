import os
import sqlite3
import streamlit as st
import pandas as pd
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hashlib
import json
#from dotenv import load_dotenv

# Load environment variables
#load_dotenv()

class SmartChatBot:
    def __init__(self, sql_db_path=None, gemini_api_key=None):
        # Use environment variables if not provided directly
        sql_db_path = sql_db_path or os.getenv('DATABASE_PATH')
        gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        
        if not sql_db_path or not gemini_api_key:
            raise ValueError("Database path and Gemini API key must be provided either as arguments or in .env file")
        
        # Gemini API setup
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # PDF embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # SQL Database setup
        self.sql_db_path = sql_db_path
        self.connection = None
        self.cursor = None
        
        # Caching directories
        self.cache_dir = '.embedding_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # PDF document storage with improved management
        self.reset_pdf_storage()
    
    def reset_pdf_storage(self):
        """Reset PDF document storage with improved structure"""
        self.pdf_docs = {
            'chunks': [],     # Store text chunks
            'metadata': [],   # Store chunk metadata
            'embeddings': []  # Store embeddings
        }
    
    def hash_text(self, text):
        """Create a consistent hash for a piece of text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding_cache_path(self, text_hash):
        """Generate cache file path for embeddings"""
        return os.path.join(self.cache_dir, f'{text_hash}_embedding.json')
    
    def cache_embedding(self, text_hash, embedding):
        """Cache embedding to file"""
        cache_path = self.get_embedding_cache_path(text_hash)
        # Convert numpy array to list for JSON serialization
        embedding_list = embedding.tolist()
        with open(cache_path, 'w') as f:
            json.dump(embedding_list, f)
    
    def load_embedding_from_cache(self, text_hash):
        """Load embedding from cache if exists"""
        cache_path = self.get_embedding_cache_path(text_hash)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return np.array(json.load(f))
        return None
    
    def advanced_pdf_chunking(self, pdf_path, chunk_size=200, overlap=50):
        """
        Improved PDF chunking strategy
        - Uses sliding window approach
        - Maintains context between chunks
        - Adds metadata for tracking
        """
        try:
            # Read the PDF
            pdf_reader = PdfReader(pdf_path)
            
            # Extract text from all pages
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"
            
            # Split text into words
            words = full_text.split()
            
            # Create chunks with overlap
            chunks = []
            metadata = []
            
            for i in range(0, len(words), chunk_size - overlap):
                # Create chunk
                chunk = ' '.join(words[i:i+chunk_size])
                
                # Generate metadata
                chunk_metadata = {
                    'source': pdf_path,
                    'start_index': i,
                    'length': len(chunk.split())
                }
                
                # Generate hash for caching
                chunk_hash = self.hash_text(chunk)
                
                # Check cache first
                cached_embedding = self.load_embedding_from_cache(chunk_hash)
                
                if cached_embedding is not None:
                    embedding = cached_embedding
                else:
                    # Generate embedding
                    embedding = self.embedding_model.encode([chunk])[0]
                    
                    # Cache the embedding
                    self.cache_embedding(chunk_hash, embedding)
                
                # Store chunk details
                chunks.append(chunk)
                metadata.append(chunk_metadata)
                
                # Store or extend embeddings
                if not self.pdf_docs['embeddings']:
                    self.pdf_docs['embeddings'] = [embedding]
                else:
                    self.pdf_docs['embeddings'].append(embedding)
            
            # Update PDF docs storage
            self.pdf_docs['chunks'].extend(chunks)
            self.pdf_docs['metadata'].extend(metadata)
            
            st.success(f"PDF {pdf_path} processed successfully! 📄")
            return len(chunks)
        
        except Exception as e:
            st.error(f"PDF processing error: {e}")
            return 0
    
    def connect_to_database(self):
        """Test database connection"""
        try:
            # Test connection without storing it
            with sqlite3.connect(self.sql_db_path) as conn:
                conn.cursor().execute("SELECT 1")
            st.success("Database connected successfully! 🎉")
            return True
        except sqlite3.Error as e:
            st.error(f"Database connection failed: {e}")
            return False
    
    def query_sql(self, natural_language_query):
        """Convert natural language to SQL query"""
        try:
            # Use Gemini to convert natural language to SQL
            prompt = f"""
            Convert this natural language query to an SQL query:
            '{natural_language_query}'
            
            Provide ONLY the SQL query, nothing else.
            """
            
            sql_response = self.gemini_model.generate_content(prompt)
            sql_query = sql_response.text.strip().replace('```sql', '').replace('```', '').strip()
            
            # Create a new connection for this query
            with sqlite3.connect(self.sql_db_path) as conn:
                # Execute the query
                df_results = pd.read_sql_query(sql_query, conn)
                
                # Log the SQL query for debugging
                st.write("Generated SQL:", sql_query)
                
                return df_results
        
        except Exception as e:
            st.error(f"SQL query error: {e}")
            st.error(f"Generated SQL query: {sql_query}")
            return None
    
    def retrieve_pdf_context(self, query, top_k=3):
        """Find most relevant PDF chunks"""
        if not self.pdf_docs['embeddings']:
            return []
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarity
        embeddings_array = np.array(self.pdf_docs['embeddings'])
        similarities = cosine_similarity([query_embedding], embeddings_array)[0]
        
        # Get top-k most similar chunks with their metadata
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_results = []
        for idx in top_indices:
            relevant_results.append({
                'chunk': self.pdf_docs['chunks'][idx],
                'metadata': self.pdf_docs['metadata'][idx],
                'similarity': similarities[idx]
            })
        
        return relevant_results
    
    def smart_chat(self, query):
        """Intelligent query routing"""
        # SQL-related keywords
        sql_keywords = ['table', 'database', 'query', 'sql', 'record', 'data']
        
        try:
            if any(keyword in query.lower() for keyword in sql_keywords):
                # SQL query route
                return self.query_sql(query)
            else:
                # PDF/RAG route
                pdf_contexts = self.retrieve_pdf_context(query)
                
                if not pdf_contexts:
                    # No PDF context, use Gemini directly
                    response = self.gemini_model.generate_content(query)
                    return response.text
                
                # Augment prompt with PDF context
                context_text = "\n".join([
                    f"Context (Similarity: {ctx['similarity']:.2f}): {ctx['chunk']}" 
                    for ctx in pdf_contexts
                ])
                
                augmented_prompt = f"""
                Context from PDFs:
                {context_text}
                
                User Query: {query}
                
                Please answer the query based on the context and your knowledge.
                Provide the most relevant and precise answer possible.
                """
                
                # Get response from Gemini
                response = self.gemini_model.generate_content(augmented_prompt)
                return response.text
        
        except Exception as e:
            st.error(f"Chat processing error: {e}")
            return "Sorry, I couldn't process your query."

def main():
    # Streamlit app configuration
    st.set_page_config(page_title="QnA Bot")
    
    # Sidebar for configuration
    st.sidebar.title("Chatbot Configuration")
    
    # Check for env configuration
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    db_path = os.getenv('DATABASE_PATH')
    
    # Warn if env vars are not set
    if not gemini_api_key:
        st.sidebar.warning("Gemini API Key not found in .env file")
        gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    
    if not db_path:
        st.sidebar.warning("Database Path not found in .env file")
        db_path = st.sidebar.text_input("Enter SQLite Database Path", 
                                        placeholder="Full path to your SQLite database")
    
    # PDF Upload
    uploaded_pdfs = st.sidebar.file_uploader("Upload PDFs", 
                                             type=['pdf'], 
                                             accept_multiple_files=True)
    
    # Main app title
    st.title("QnA Bot")
    
    # Initialize or reset chatbot
    if 'chatbot' not in st.session_state or st.sidebar.button("Reset Chatbot"):
        if gemini_api_key and db_path:
            try:
                st.session_state.chatbot = SmartChatBot(db_path, gemini_api_key)
                st.session_state.chatbot.connect_to_database()
            except ValueError as e:
                st.error(str(e))
        else:
            st.warning("Please provide API Key and Database Path")
    
    # PDF Processing
    if uploaded_pdfs and 'chatbot' in st.session_state:
        for pdf in uploaded_pdfs:
            # Save uploaded PDF temporarily
            with open(pdf.name, 'wb') as f:
                f.write(pdf.getvalue())
            
            # Process PDF
            st.session_state.chatbot.advanced_pdf_chunking(pdf.name)
            
            # Optional: Remove temporary file
            os.remove(pdf.name)
    
    # Chat interface
    if 'chatbot' in st.session_state:
        # Chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get chatbot response
            response = st.session_state.chatbot.smart_chat(prompt)
            
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
    else:
        st.info("Configure the chatbot in the sidebar to get started!")

if __name__ == '__main__':
    main()
