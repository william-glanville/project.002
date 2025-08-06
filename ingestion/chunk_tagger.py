import json
import constants
import random
from transformers import pipeline

class ChunkTagger:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model=constants.MODEL_CHUNK_TAGGER_NAME)
        self.topics = ["Linear Algebra", "Ethics in AI", "Probability", "Neural Networks"]
        self.difficulties = ["easy", "medium", "hard"]

    def tag_topic(self, text):
        result = self.classifier(text, self.topics)
        return result["labels"][0]

    def tag_difficulty(self, text):
        # Simple heuristic: sentence length or complexity
        length = len(text.split())
        if length < 50:
            return "easy"
        elif length < 100:
            return "medium"
        else:
            return "hard"

    def tag_chunk(self, chunk):
        return {
            **chunk,
            "topic_tags": json.dumps([self.tag_topic(chunk["text"])]),
            "difficulty": self.tag_difficulty(chunk["text"])
        }

    def tag_chunks(self, chunks):
        return [self.tag_chunk(chunk) for chunk in chunks]