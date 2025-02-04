from transformers import pipeline

class EntityClassifier:
    def __init__(self, candidate_labels = ["organization","occupation"], entities = []):
        self.__classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",
            device=-1  # CPU
        )
        self.__candidate_labels = candidate_labels
        self.entities = entities
        self.results = {}

    def update_candidate_labels(self, candidate_labels):
        if self.__candidate_labels!=candidate_labels:
            self.__candidate_labels = candidate_labels
            self.results = {} # clear all results

    def load_entities(self, entities):
        self.entities = entities

    def clear_results(self):
        to_pop = set(self.results.keys()) - set(self.entities)
        for x in to_pop:
            self.results.pop(x)

    def classify_bulk(self, candidate_labels=[], entities=[]):
        if len(candidate_labels)>0:
            self.update_candidate_labels(candidate_labels)
        if len(entities)>0:
            self.load_entities(entities)
        new_entities = list(set(self.entities)-set(self.results.keys()))
        classified_outputs = self.__classifier(
            new_entities,
            candidate_labels=self.__candidate_labels,
            hypothesis_template="This is a {}."
        )
        self.results = self.results | {new_entities[i]:(classified_outputs[i]['labels'][0],classified_outputs[i]['scores'][0]) for i in range(len(new_entities))}
        return {x:self.results[x] for x in self.entities}