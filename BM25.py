import math
class BM25:
    """
    Best Match 25.

    Parameters to tune
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document. So [{'hi': 1}] means
        the first document contains the term 'hi' 1 time.
        The frequnecy is normilzied by the max term frequency for each document.

    doc_len_ : dict[int]
        Number of terms per document. So [3] = 10 means the
        document 3 contains 10 terms.

    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.

    avg_doc_len_ : float
        Average number of terms for documents in the corpus.

    idf_ : dict[str, float]
        Inverse Document Frequency per term.
    """

    def __init__(self,doc_len,df,N,total_terms,k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.doc_len_ = doc_len
        self.df_ = df
        self.N_ = N
        self.avgdl_ = total_terms / N

    def calc_idf(self, query):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        # YOUR CODE HERE
        # i create the idf dict
        idf = {}
        # i go over each term
        for term in query:
            # i get the number of docs containing that term (0 if none)
            num_of_docs_containing_term = self.df_.get(term, 0)
            # i calculate the the idf value of that term using the idf equation provided in the beginning of the task.
            idf[term] = math.log( (self.N_ - num_of_docs_containing_term + 0.5) / (num_of_docs_containing_term + 0.5)) + 1
        return idf

    def score_term(self, tf, doc_id, idf):
        """
        BM25 contribution of a single term to a single document.
        """
        dl = self.doc_len_.get(doc_id, 0)
        if dl == 0:
            return 0.0

        B = 1 - self.b + self.b * (dl / self.avgdl_)
        return idf * ((tf * (self.k1 + 1)) / (tf + self.k1 * B))

