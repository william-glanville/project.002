import re
import requests
import logging
import time
import json
import hashlib
import shelve


from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant that generates reference sentences for a given keyword in a given context."
PROMPT = "Generate {n} informative sentences about {keyword} in the context of {context} that is suitable for educational or technical contexts. Return the result as a JSON array of strings."

class ReferenceCache:
    def __init__(self, persistent=False, path="reference_cache.db"):
        self.persistent = persistent
        self.path = path
        self._cache = shelve.open(path) if persistent else {}

    def _serialize_key(self, chunks):
        serialized = json.dumps(chunks, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def get(self, chunks):
        key = self._serialize_key(chunks)
        return self._cache.get(key)

    def set(self, chunks, references):
        key = self._serialize_key(chunks)
        self._cache[key] = references

    def contains(self, chunks):
        key = self._serialize_key(chunks)
        return key in self._cache

    def close(self):
        if self.persistent:
            self._cache.close()

class LocalLLMClient:
    def __init__(self, endpoint="http://192.168.0.128:1234/v1/chat/completions", model_id="openai/gpt-oss-20b", headers=None, timeout=30):
        self.endpoint = endpoint
        self.model_id = model_id
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout

    def generate(self, prompt, max_tokens=256, temperature=0.7, retries=3):
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        for attempt in range(retries):
            try:
                response = requests.post(self.endpoint, json=payload, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()

                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    raise ValueError("Unexpected response format from LMStudio")

            except Exception as e:
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {str(e)}")
                time.sleep(1)

        raise RuntimeError("Failed to get response from LMStudio after retries")


class ReferenceBuilder:
    def __init__(self, max_keywords=10, sentences_per_keyword=2, cache=None):
        self.max_keywords = max_keywords
        self.sentences_per_keyword = sentences_per_keyword
        self.llm = LocalLLMClient()
        self.parser = OutputParser(mode="json")
        self.cache = cache or ReferenceCache()

    def _hash_chunks(self, chunks):
        serialized = json.dumps(chunks, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def extract_keywords(self, chunks):
        texts = [chunk["text"] for chunk in chunks if chunk["text"].strip()]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_keywords)
        X = vectorizer.fit_transform(texts)
        keywords = vectorizer.get_feature_names_out()
        logger.info(f"Extracted keywords: {keywords}")
        return keywords

    def generate_reference_sentences(self, keywords, context):
        references = []
        for kw in keywords:
            prompt = PROMPT.format(n=self.sentences_per_keyword, keyword=kw, context=context )
            try:
                response = self.llm.generate(prompt)
                sentences = self._split_sentences(response)
                references.extend(sentences)
            except Exception as e:
                logger.warning(f"LLM failed for keyword '{kw}' in context '{context}': {str(e)}")
        return references

    def _split_sentences(self, text):
        return self.parser.parse(text)

    def build_reference_set(self, chunks, context):
        if self.cache.contains(chunks):
            logger.info("Using cached references")
            return self.cache.get(chunks)

        keywords = self.extract_keywords(chunks)
        references = self.generate_reference_sentences(keywords, context)
        self.cache.set(chunks, references)
        logger.info(f"Generated {len(references)} {context} reference sentences")
        return references


class OutputParser:
    def __init__(self, mode="auto"):
        """
        mode: 'json', 'lines', or 'auto'
        """
        self.mode = mode

    def parse(self, text):
        """
        Main entry point to parse LLM output.
        Returns a list of cleaned strings.
        """
        if self.mode == "json":
            return self._parse_json(text)
        elif self.mode == "lines":
            return self._parse_lines(text)
        else:  # auto
            parsed = self._parse_json(text)
            if parsed:
                return parsed
            return self._parse_lines(text)

    def _parse_json(self, text):
        """
        Extracts and parses a JSON array from LLM output.
        Handles markdown-style code blocks and escaped characters.
        """
        match = re.search(r"```(?:json)?\s*(\[[^\]]*\])\s*```", text, re.DOTALL)
        if not match:
            match = re.search(r"(\[[^\]]*\])", text, re.DOTALL)
        if match:
            raw_json = match.group(1)
            try:
                cleaned = raw_json.replace("\\n", " ").replace("\\", "")
                data = json.loads(cleaned)
                return [self._normalize(s) for s in data if isinstance(s, str) and s.strip()]
            except json.JSONDecodeError:
                logger.warning("Failed to decode JSON array.")
        return None

    def _parse_lines(self, text):
        """
        Splits text by lines and normalizes each line.
        """
        lines = text.splitlines()
        return [self._normalize(line) for line in lines if line.strip()]

    def _normalize(self, s):
        """
        Cleans up whitespace and formatting.
        """
        return re.sub(r"\s+", " ", s).strip()

def main():
    builder = ReferenceBuilder(sentences_per_keyword=3)
    keywords = ["Parallelism", "Divide‑and‑Conquer", "Big O Complexity", "Randomized Algorithms", "Dynamic Programming", "Greedy Strategies", "Approximation Ratios", "NP‑Hardness", "Heuristic Search", "Graph Traversal"]
    context = "Computing Algorithmics"
    references = builder.generate_reference_sentences(keywords, context)
    print(references)

if __name__ == "__main__":
    main()