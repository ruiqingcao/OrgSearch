from transformers import pipeline
import math

class EntityClassifier:
    def __init__(self, entities = [], candidate_labels = ["organization","occupation"], batchsize=100):
        self.__classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",
            device=-1  # CPU
        )
        self.__candidate_labels = candidate_labels
        self.__batchsize = batchsize
        self.entities = entities
        self.results = {}

    def update_candidate_labels(self, candidate_labels):
        if self.__candidate_labels!=candidate_labels:
            self.__candidate_labels = candidate_labels
            self.results = {} # clear all results

    def update_batchsize(self, batchsize):
        self.__batchsize = batchsize

    def load_entities(self, entities):
        self.entities = entities

    def clear_results(self):
        to_pop = set(self.results.keys()) - set(self.entities)
        for x in to_pop:
            self.results.pop(x)

    def classify_bulk(self, entities=[], candidate_labels=[], batchsize=None):
        if len(candidate_labels)>0:
            self.update_candidate_labels(candidate_labels)
        if len(entities)>0:
            self.load_entities(entities)
        if batchsize is not None:
            self.__batchsize = batchsize
        new_entities = list(set(self.entities)-set(self.results.keys()))
        n_batches = math.floor((len(new_entities)-1)/self.__batchsize)+1
        for n in range(n_batches):
            batch = new_entities[n*self.__batchsize:(n+1)*self.__batchsize]
            classified_outputs = self.__classifier(
                batch,
                candidate_labels=self.__candidate_labels,
                hypothesis_template="This is a {}."
            )
            self.results = self.results | {batch[i]:(classified_outputs[i]['labels'][0],classified_outputs[i]['scores'][0]) for i in range(len(batch))}
        return {x:self.results[x] for x in self.entities}