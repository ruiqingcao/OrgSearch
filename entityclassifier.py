from transformers import pipeline
import math

class EntityClassifier:
    def __init__(self, entities = [], candidate_labels = ["organization","occupation"], batchsize=100, threshold=0.7):
        self.__classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",
            device=-1  # CPU
        )
        self.__candidate_labels = candidate_labels
        self.__batchsize = batchsize
        self.__threshold = threshold
        if len(entities)==0:
            self.entities = entities
            self.entities_cleaned = []
            self.raw_to_index = {}
        else:
            self.load_entities(entities)
        self.results = {}

    def clean_string(self,st):
        return st.split('/')[0].title()

    def update_candidate_labels(self, candidate_labels):
        if self.__candidate_labels!=candidate_labels:
            self.__candidate_labels = candidate_labels
            self.results = {} # clear all results

    def update_threshold(self, threshold):
        self.__threshold = threshold

    def update_batchsize(self, batchsize):
        self.__batchsize = batchsize

    def load_entities(self, entities):
        self.entities = entities
        self.entities_cleaned = [self.clean_string(entity) for entity in entities]
        self.raw_to_index = {self.entities[i]:i for i in range(len(self.entities))}

    def clear_results(self):
        to_pop = set(self.results.keys()) - set(self.entities)
        for x in to_pop:
            self.results.pop(x)

    def classify_bulk(self, entities=[], candidate_labels=[], batchsize=None, threshold=None):
        if len(candidate_labels)>0:
            self.update_candidate_labels(candidate_labels)
        if len(entities)>0:
            self.load_entities(entities)
        if batchsize is not None:
            self.update_batchsize(batchsize)
        if threshold is not None:
            self.update_threshold(threshold)
        new_entities = list(set(self.entities)-set(self.results.keys()))
        n_batches = math.floor((len(new_entities)-1)/self.__batchsize)+1
        for n in range(n_batches):
            batch = new_entities[n*self.__batchsize:(n+1)*self.__batchsize]
            batch_cleaned = [self.entities_cleaned[self.raw_to_index[x]] for x in batch]
            classified_outputs = self.__classifier(
                batch_cleaned,
                candidate_labels=self.__candidate_labels,
                hypothesis_template="This is a {}."
            )
            self.results = self.results | {batch[i]:list(zip(classified_outputs[i]['labels'],classified_outputs[i]['scores'])) for i in range(len(batch))}
        return {x:[tu for tu in self.results[x] if tu[1]>=self.__threshold] for x in self.entities}

    def retrieve_results(self, threshold=None):
        if threshold is not None:
            self.update_threshold(threshold)
        return {x:[tu for tu in self.results[x] if tu[1]>=self.__threshold] for x in self.results.keys()}

