# Search Engine 

## Project Overview

This project implements a large scale search engine over the English Wikipedia corpus.  
Given a user query, the system retrieves and ranks relevant documents using a combination of BM25 scoring, PageRank, and page view statistics.

The search engine is exposed through a Flask REST API and runs on Google Cloud Platform, where all indices data are stored.

A typical query is issued via:

http://34.10.188.147:8080/search?query=your+query+here

The system returns the top 100 most relevant Wikipedia articles, where each result contains a document ID and its corresponding title.

---

## Project Structure
search_frontend.py # Flask server and main query logic

inverted_index_gcp.py # Inverted index implementation

BM25.py # BM25 ranking model

text_Modification.py # Tokenization, stopwords, stemming

---
## Code Organization and Main Components

### search_frontend.py

This file is where everything comes together.  
It runs the Flask server and handles the full search process from the moment the user sends a query until results are returned.

What we do in this file:
- Start the Flask application.
- Connect to our Google Cloud bucket.
- Load all the data we need:
  - Body inverted index
  - Title inverted index
  - PageRank values
  - Page views data
  - Document ID to title mapping
- Handle user requests and return results.

The main function in that class is the "search" below is how it runs.
- We get the query from the user.
- We tokenize it, remove stopwords, and apply stemming.
- For each query term, we calculate BM25 scores:
  - Once using the body index
  - Once using the title index
- We combine the scores (75% body, 25% title).
- We add normalized  PageRank and page views to improve precision.
- Finally, we return the top 100 documents as (doc_id, title).

To make things faster, we use threading so posting lists from the body and title indices are read in parallel.

---

### inverted_index_gcp.py

This file contains our inverted index implementation.

What this file does:
- Stores important information about the corpus like:
  - Number of documents
  - Document lengths
  - Document frequency per term
  - Total number of unique terms
  - term frequency per term
- Stores posting lists on disk or in Google Cloud Storage.
- Reads posting lists only when needed during search.

Main parts:
- InvertedIndex
  - Holds all corpus statistics.
  - Keeps track of where each posting list is saved.
  - Allows reading posting lists from storage.
- MultiFileWriter
  - Writes posting lists into multiple binary files.
- MultiFileReader
  - Reads posting lists from multiple files efficiently.

This setup allows us to work with very large data without loading everything into memory.

---

### BM25.py

This file contains the BM25 ranking logic.

What this file does:
- Calculates IDF values for query terms.
- Computes how much each term contributes to a document score.

Main functions:
- calc_idf(query_terms)
  - Calculates IDF for every term in the query.
- score_term(tf, doc_id, idf)
  - Calculates the BM25 score of a term in a document.

We use two BM25 objects:
- One for the body index
- One for the title index

This lets us tune each one separately. 

---

### text_Modification.py

This file handles text preprocessing.

What this file does:
- Tokenizes text using a regex.
- Removes stopwords.
- Applies Porter stemming.

We use the same preprocessing both when building the index and when processing queries.

---

## Data Sources

All large data files are stored in **Google Cloud Storage**, including:
- Body and title inverted indices
- PageRank data
- Page views data
- Document ID to title mapping

The data is loaded once when the server starts and reused for every query.

