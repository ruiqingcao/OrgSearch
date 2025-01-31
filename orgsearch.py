import numpy as np
import pandas as pd
import string
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util


class orgsearch:
    """
    This class expresses a data object that allows the user to find the best matches from an organizations name database close to a query name 
    """

    def __init__(self, orgnames=[], orgnames_clean=[], cleaned_to_index = {}, queries=None, results={}, topk=1, threshold_keep = [90,10], fuzz_methods=[fuzz.token_set_ratio,fuzz.token_sort_ratio], tf=True):
        """
        Constructor method to initialize instance attributes.
        Args:
            orgnames: List of organizations in the database to compare the query with
            queries: Queries (or query) to find the closest organization in the database
            results: The stored results that maps a query to an output list of top matches (for the query relative to the organizations database)
            topk: Number of top matches to return
        """
        # Instance attributes
        self.__orgnames = orgnames
        self.__orgnames_clean = orgnames_clean
        self.__cleaned_to_index = cleaned_to_index
        self.queries = queries
        self.results = results
        self.__topk = topk
        self.__threshold_keep = threshold_keep
        self.__fuzz_methods = fuzz_methods
        self.__tf = tf
        if tf:
            self.__tfmodel = SentenceTransformer('all-MiniLM-L6-v2')  # load a pre-trained similarity model: Fast & accurate
            self.__embeddings = dict()

    
    def clean_string(self,st):
        """
        Remove punctuations from a string and convert it to lower case.
        """
        return ''.join(c.lower() for c in st if c not in string.punctuation)

    
    def load_comparison_database(self,file_or_list,varname=None):
        """
        Load the organization database.
        """
        if varname==None:
            orgnames = sorted(list(file_or_list))
        else:
            orgnames = set(pd.read_csv(file_or_list,dtype={varname:str})[varname].unique())
            orgnames = sorted(list(orgnames))            
        self.__orgnames = orgnames
        self.__orgnames_clean = [self.clean_string(name) for name in orgnames]
        self.__cleaned_to_index = {self.__orgnames_clean[i]:i for i in range(len(self.__orgnames_clean))}
        self.queries = None
        self.results = {}

    def update_topk(self,topk_new):
        self.__topk = topk_new
        self.queries = None
        self.results = {}

    def update_threshold(self,threshold_new):
        self.__threshold_keep = threshold_new
        self.queries = None
        self.results = {}

    def update_fuzz_methods(self,fuzz_methods_new):
        self.__fuzz_methods = fuzz_methods_new
        self.queries = None
        self.results = {}

    
    def load_queries(self,queries):
        """
        Load the queries (or query).
        """
        self.queries = queries
        

    def compare_thres(self, scr):
        """
        Judge if scr exceeds the values in self.__threshold_keep.
        """
        if type(self.__threshold_keep)==list:
            for i in range(len(self.__threshold_keep)):
                if self.__threshold_keep[i]>scr[i]:
                    return False
        else:
            if self.__threshold_keep>scr[0]:
                return False
        return True


    def transformer_similarity(self,q1,q2):
        if q1 not in self.__embeddings.keys():
            self.__embeddings[q1] = self.__tfmodel.encode(q1, convert_to_tensor=True)
        if q2 not in self.__embeddings.keys():
            self.__embeddings[q2] = self.__tfmodel.encode(q2, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(self.__embeddings[q1], self.__embeddings[q2]).item()
        return similarity_score



    def match_query(self,query):
        """
        Find the top matches for one query.
        """
        if query in self.results.keys():
            return self.results[query]
        q = self.clean_string(query)
        scores = dict()
        score_keys = []
        scores = [[self.__cleaned_to_index[name],tuple([f(q,name) for f in self.__fuzz_methods])] for name in self.__orgnames_clean]
        scores.sort(key=lambda x: x[1], reverse=True)
        if self.__tf:
            matches = [(query,self.__orgnames[score[0]],score[1],self.transformer_similarity(query,self.__orgnames[score[0]])) for score in scores[:self.__topk] if self.compare_thres(score[1])]
        else:
            matches = [(query,self.__orgnames[score[0]],score[1]) for score in scores[:self.__topk] if self.compare_thres(score[1])]
        self.results[query] = matches
        #print(matches)
        return matches
        
        
    def get_top_matches(self,queries=None,topk=None,threshold_keep=None, fuzz_methods=None):
        if queries!=None:
            self.load_queries(queries)
        if (topk!=None) & (topk!=self.__topk):
            self.update_topk(topk)
        if (threshold_keep!=None) & (threshold_keep!=self.__threshold_keep):
            self.update_threshold(threshold_keep)
        if (fuzz_methods!=None) & (fuzz_methods!=self.__fuzz_methods):
            self.update_fuzz_methods(fuzz_methods)
        if type(self.queries)==str:
            return self.match_query(queries)
        else:
            return [self.match_query(query) for query in queries]

    
