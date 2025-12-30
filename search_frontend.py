import math
import os
from flask import Flask, request, jsonify
import pickle
from text_Modification import all_stopwords,RE_WORD,ps
from collections import defaultdict
from BM25 import BM25
from google.cloud import storage
import gzip
from inverted_index_gcp import InvertedIndex
from concurrent.futures import ThreadPoolExecutor
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\asust\Desktop\search_engine_git\Search_Engine\uni-project-480107-d6cc24a4a250.json"
BUCKET_NAME = "214682189"
TITLE_DIR = "title_index"
TITLE_INDEX = "title"
BODY_INDEX = "body"
BODY_DIR = "body_index"
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
print("Loading Body Index...")
body_index = InvertedIndex.read_index(BODY_DIR, BODY_INDEX, BUCKET_NAME)
print("Loading Title Index...")
title_index = InvertedIndex.read_index(TITLE_DIR, TITLE_INDEX, BUCKET_NAME)
print("Loading Page Views...")
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
blob_pv = bucket.blob("page_views_august_2021.pkl")
with blob_pv.open("rb") as f:
    page_views = pickle.load(f)
print("Loading PageRank...")
pagerank_scores = defaultdict(float)
pr_path = "pr/part-00000-dfa568ba-d8f3-4828-9ded-c144a863ddec-c000.csv.gz"
blob_pr = bucket.blob(pr_path)
with blob_pr.open("rb") as f:
    with gzip.open(f, "rt") as gz:
        for line in gz:
            try:
                doc_id, rank = line.strip().split(',')
                pagerank_scores[int(doc_id)] = float(rank)
            except ValueError:
                continue

bm25_body = BM25(doc_len=body_index.doc_len, df=body_index.document_frequencey_per_term, N=body_index.N, total_terms=body_index.total_corpus_terms)
bm25_title = BM25(doc_len=title_index.doc_len,df=title_index.document_frequencey_per_term, N=title_index.N,total_terms=title_index.total_corpus_terms)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)


    tokens = [ps.stem(m.group()) for m in RE_WORD.finditer(query.lower()) if m.group() not in all_stopwords]


    body_index_idf = bm25_body.calc_idf(tokens)
    title_index_idf = bm25_title.calc_idf(tokens)

    body_index_scores = defaultdict(float)
    title_index_scores = defaultdict(float)


    with ThreadPoolExecutor() as executor:

        body_threads = [executor.submit(read_posting, body_index, term, BODY_DIR) for term in tokens]
        title_threads = [executor.submit(read_posting, title_index, term, TITLE_DIR) for term in tokens]


        for thread in body_threads:
            term, posting_list = thread.result()
            term_idf = body_index_idf.get(term, 0)
            for doc_id, tf in posting_list:
                body_index_scores[doc_id] += bm25_body.score_term(tf, doc_id, term_idf)


        for thread in title_threads:
            term, posting_list = thread.result()
            term_idf = title_index_idf.get(term, 0)
            for doc_id, tf in posting_list:
                title_index_scores[doc_id] += bm25_title.score_term(tf, doc_id, term_idf)


    final_score = {}
    candidate_docs = set(body_index_scores.keys()) | set(title_index_scores.keys())

    page_views_tuner = 1.4
    page_rank_tuner = 0.8

    for doc_id in candidate_docs:
        fused_bm25 = (0.50 * body_index_scores.get(doc_id, 0)) + (0.50 * title_index_scores.get(doc_id, 0))


        document_views = page_views.get(doc_id, 0)
        views_score = math.log10(document_views + 1)

        document_page_rank = pagerank_scores.get(doc_id, 0)
        page_rank_score = math.log10(document_page_rank + 1)

        final_score[doc_id] = fused_bm25 + (page_views_tuner * views_score) + (page_rank_tuner * page_rank_score)

    res = sorted(final_score.items(), key=lambda item: item[1], reverse=True)[:100]

    # Return formatted list
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for wiki_id in wiki_ids:
        res.append(page_views.get(wiki_id,0))
    # END SOLUTION
    return jsonify(res)
def read_posting(index, term, dir_name):
    if term not in index.posting_locs:
        return term, []
    posting_list = index.read_a_posting_list(base_dir=dir_name, w=term, bucket_name=BUCKET_NAME)
    return term, posting_list
def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
