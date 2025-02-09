
import numpy as np
import pandas as pd
import string
import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util


class OrgSearch:
    """
    This class defines a data object that finds the closest matching organization names from a large database based on a given set of input queries.
    """
    def __init__(self, orgnames=[], clean=True, queries=None, fuzz_threshold=[90,10], fuzz_topk=1, tf_on=True, tf_threshold=0.7, tf_topk=1):
        if len(orgnames)==0:
            self.__orgnames = []
            self.__orgnames_clean = []
            self.__cleaned_to_index = {}
        else:
            self.load_comparison_database(file_or_list=orgnames, clean=clean)
        self.queries = queries
        self.results = {}
        self.__fuzz_methods = [fuzz.token_set_ratio,fuzz.token_sort_ratio]
        self.__fuzz_threshold = fuzz_threshold
        self.__fuzz_topk = fuzz_topk
        if tf_on:
            self.__tf_on = True
            self.__tf_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')  # load pre-trained similarity model
            self.__embeddings = dict()
            self.__tf_threshold = tf_threshold
            self.__tf_topk = tf_topk
        else:
            self.__tf_on = False
            self.__tf_model = None
            self.__embeddings = None
            self.__tf_threshold = None
            self.__tf_topk = None
    

    def clean_string(self,st,clean):
        """
        Remove punctuations and convert to lower case.
        """
        if clean:
            st_split = st.split('/')
            for item in st_split:
                if item!='':
                    punc = re.sub(r'[-]','',string.punctuation)
                    return ''.join(c.lower() for c in item if c not in punc)
        else:
            return st.lower()


    def load_comparison_database(self,file_or_list,varname=None,clean=True):
        """
        Load the large database to query against.
        """
        if varname==None:
            orgnames = set(file_or_list)
            orgnames.discard(np.nan)
            orgnames.discard(None)
            orgnames = sorted(list(orgnames))
        else:
            orgnames = set(pd.read_csv(file_or_list,dtype={varname:str})[varname].unique())
            orgnames.discard(np.nan)
            orgnames.discard(None)
            orgnames = sorted(list(orgnames))
        orgnames = [x for x in orgnames if x[0] not in string.punctuation]
        orgnames = [x for x in orgnames if x[-1] not in '"\'(),-./:;<=>?@[\\]^_`{|}~']
        self.__orgnames = orgnames
        self.__orgnames_clean = [self.clean_string(name,clean) for name in orgnames]
        self.__cleaned_to_index = {self.__orgnames_clean[i]:i for i in range(len(self.__orgnames_clean))}
        self.results = {}


    def update_fuzz_params(self, topk=None, threshold=None, methods=None):
        """
        Update fuzzy matching parameters and reset results to empty.
        """
        if topk is not None:
            self.__fuzz_topk = topk
        if threshold is not None:
            self.__fuzz_threshold = threshold
        if methods is not None:
            self.__fuzz_methods = methods
        self.results = {}
    

    def update_tf_params(self, tf_on=None, topk=None, threshold=None, model=None):
        """
        Update transformer parameters and reset results to empty.
        """
        if tf_on is not None:
            self.__tf_on = tf_on
        if topk is not None:
            self.__tf_topk = topk
        if threshold is not None:
            self.__tf_threshold = threshold
        if model is not None:
            self.__tf_model = model
        self.results = {}


    def load_queries(self,queries):
        """
        Load queries (or a query).
        """
        self.queries = queries


    def clear_results(self):
        if type(self.queries)==str:
            to_pop = set(self.results.keys()) - {self.queries}
        else:
            to_pop = set(self.results.keys()) - set(self.queries)
        for x in to_pop:
            self.results.pop(x)
        

    def compare_fuzz_threshold(self, scr):
        """
        Return True if scr passes fuzzy matching scores threshold self.__fuzz_threshold, otherwise return False.
        """
        if type(self.__fuzz_threshold)==list:
            for i in range(len(self.__fuzz_threshold)):
                if scr[i]<self.__fuzz_threshold[i]:
                    return False
        elif type(scr)==list:
            if scr[0]<self.__fuzz_threshold:
                return False
        elif scr<self.__fuzz_threshold:
            return False
        return True


    def compare_tf_threshold(self, scr):
        """
        Return True if scr passes transformer similarity score threshold self.__tf_threshold, otherwise return False.
        """
        if type(self.__tf_threshold)==list:
            for i in range(len(self.__tf_threshold)):
                if scr[i]<self.__tf_threshold[i]:
                    return False
        elif type(scr)==list:
            if scr[0]<self.__tf_threshold:
                return False
        elif scr<self.__tf_threshold:
            return False
        return True


    def transformer_similarity(self,q1,q2):
        if q1 not in self.__embeddings.keys():
            self.__embeddings[q1] = self.__tf_model.encode(q1, convert_to_tensor=True)
        if q2 not in self.__embeddings.keys():
            self.__embeddings[q2] = self.__tf_model.encode(q2, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(self.__embeddings[q1], self.__embeddings[q2]).item()
        return similarity_score


    def match_query(self,query):
        """
        Find the closest matches for a single query.
        """
        if query in self.results.keys():
            return self.results[query]
        q = self.clean_string(query,clean=True)
        try:
            ind = self.__orgnames_clean.index(q)
            if self.__tf_on:
                matches = [(query,self.__orgnames[ind],(100,100),None)]
            else:
                matches = [(query,self.__orgnames[ind],(100,100))]
            self.results[query] = matches
            return matches
        except:
            None
        fuzz_scores = [[self.__cleaned_to_index[name],tuple([f(q,name) for f in self.__fuzz_methods])] for name in self.__orgnames_clean] # a list of [index, tuple of scores from the fuzz algos]
        fuzz_scores.sort(key=lambda x: x[1], reverse=True) # sort the list of scores in lexicographic order (of the fuzz method scores) from largest to smallest
        fuzz_scores = [score for score in fuzz_scores[:self.__fuzz_topk] if self.compare_fuzz_threshold(score[1])] # retain only the scores in the fuzz_topk ranks that are not lower than fuzz_thresholds         
        if self.__tf_on:
            fuzz_tf_scores = [[score[0],score[1],self.transformer_similarity(query,self.__orgnames[score[0]])] for score in fuzz_scores] # a list of [index, tuple of scores from the fuzz algos, score from the transformer model]
            fuzz_tf_scores.sort(key=lambda x: x[2], reverse=True) # sort the list of scores in lexocographic order (of the transformer model scores) from largest to smallest
            fuzz_tf_scores = [score for score in fuzz_tf_scores[:self.__tf_topk] if self.compare_tf_threshold(score[2])] # retain only the scores in the tf_topk ranks that are not lower than tf_thresholds
            matches = [(query,self.__orgnames[score[0]],score[1],score[2]) for score in fuzz_tf_scores]
        else:
            matches = [(query,self.__orgnames[score[0]],score[1]) for score in fuzz_scores]
        self.results[query] = matches
        #print(matches) # run if in debug mode
        return matches

        
    def get_top_matches(self, queries=None, fuzz_topk=None, fuzz_threshold=None, fuzz_methods=None, tf_on=None, tf_topk=None, tf_threshold=None, tf_model=None):
        """
        Find the closest matches for a single query or a list of queries.
        If parameters are provided as arguments, then update the object attributes with these new parameters before running the matching algorithms.
        """
        if queries!=None:
            self.load_queries(queries)
        if ((fuzz_topk!=None) & (fuzz_topk!=self.__fuzz_topk)) | ((fuzz_threshold!=None) & (fuzz_threshold!=self.__fuzz_threshold)) | ((fuzz_methods!=None) & (fuzz_methods!=self.__fuzz_methods)):
            self.update_fuzz_params(topk=fuzz_topk, threshold=fuzz_threshold, methods=fuzz_methods)
        if ((tf_on!=None) & (tf_on!=self.__tf_on)) | ((tf_topk!=None) & (tf_topk!=self.__tf_topk)) | ((tf_threshold!=None) & (tf_threshold!=self.__tf_threshold)) | ((tf_model!=None) & (tf_model!=self.__tf_model)):
            self.update_tf_params(tf_on=tf_on, topk=tf_topk, threshold=tf_threshold, model=tf_model)
        if type(self.queries)==str:
            return self.match_query(self.queries)
        else:
            return [self.match_query(query) for query in self.queries]

    
