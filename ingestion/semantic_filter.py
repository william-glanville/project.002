import re
import constants

from sentence_transformers import SentenceTransformer, util

class SemanticFilter:
    def __init__(self, logger, model_name=constants.MODEL_EMBED_NAME, threshold=0.3):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.logger = logger

    def is_low_information(self, text):
        # Heuristic filters
        if len(text.strip()) < 30:
            return True
        if re.match(r'^\s*(Page\s*\d+|Chapter\s*\d+)?\s*$', text, re.IGNORECASE):
            return True
        return False

    def semantic_score(self, text, reference_texts):
        text_embedding = self.model.encode(text, convert_to_tensor=True)
        ref_embeddings = self.model.encode(reference_texts, convert_to_tensor=True)
        scores = util.cos_sim(text_embedding, ref_embeddings)
        return scores.max().item()

    def filter_chunks(self, chunks, reference_texts):
        filtered = []
        for chunk in chunks:
            clean_text = self.normalize_text(chunk["text"])
            if self.is_low_information(clean_text):
                continue
            score = self.semantic_score(clean_text, reference_texts)
            if score >= self.threshold:
                filtered.append(chunk)
        self.logger.info(f"Filtered out {len(chunks) - len(filtered)} low-information chunks")
        return filtered

    def normalize_text(self, text):
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
        return text.strip()
