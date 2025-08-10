import re
import requests
import logging
import time
import json
import hashlib
import shelve
import constants
import spacy
import ftfy
import wordninja
from collections import defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

nlp = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = "Summarize the following document as a concise statement that can be used as a title or subject."
REFERENCE_PROMPT = "You are a helpful assistant that generates reference sentences for a given keyword in a given context."
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

    def generate(self, prompt, system_prompt, max_tokens=256, temperature=0.7, retries=3):
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
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



class SubjectExtractor:
    def __init__(self, llm_client, max_chunks=10):
        self.llm = llm_client
        self.max_chunks = max_chunks

    def extract_subject(self, chunks):
        sample_texts = [chunk["text"] for chunk in chunks[:self.max_chunks] if chunk["text"].strip()]
        combined = "\n".join(sample_texts)

        prompt = (
            "Based on the following document excerpts, summarize the main subject or theme in one concise statement. Prefix the final title/subject with 'TITLE_IS_:'\n\n"
            f"{combined}\n\nSubject:"
        )

        try:
            subject = self.llm.generate(prompt, SUMMARY_PROMPT, max_tokens=50)
            logger.info(f"Proposed subject response: {constants.safe_text(subject)}")
            #match = re.search(r"\*\*(.*?)\*\*", subject)
            match = re.search(r"TITLE_IS_:\s*(.+)", subject)
            result = match.group(1) if match else subject
            logger.info(f"Extracted subject: {constants.safe_text(result)}")
            return result
        except Exception as e:
            logger.warning(f"Failed to extract subject context: {str(e)}")
            return "General Document"

class ReferenceBuilder:
    def __init__(self, llm_client, max_keywords=10, sentences_per_keyword=2, cache=None, stopwords=None):
        self.max_keywords = max_keywords
        self.sentences_per_keyword = sentences_per_keyword
        self.llm = llm_client
        self.parser = OutputParser(mode="json")
        self.cache = cache or ReferenceCache()
        self.stopwords = set(stopwords or [])

    def _hash_chunks(self, chunks):
        serialized = json.dumps(chunks, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    def _split_text_with_wordninja(self, text_block):
        """
        Splits a block of text into individual words using wordninja.
        Handles compound words and removes excessive whitespace.

        Args:
            text_block (str): The input string to process.

        Returns:
            str: A space-separated string of segmented words.
        """
        # Normalize whitespace and split into tokens
        tokens = text_block.strip().split()

        # Apply wordninja to each token
        segmented = []
        for token in tokens:
            words = wordninja.split(token)
            segmented.append(" ".join(words))

        # Join all segmented tokens into a single string
        return " ".join(segmented)

    def _extract_concepts(self, text):
        candidate = text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else text
        candidate = self._split_text_with_wordninja(candidate)
        doc = nlp(candidate)
        concepts = ConceptSet()
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:
                word = token.text.lower()
                if (( word in constants.CONCEPT_WHITELIST or
                    (word not in constants.CONCEPT_BLACKLIST and len(word) > 3) or
                    (not any(char.isdigit() for char in word)) or
                    ((word not in self.stopwords) and (word not in ENGLISH_STOP_WORDS))) and
                    self._is_grammatical(word)
                ):
                    concepts.add(word)
        return concepts.get_all()

    def _is_grammatical(self, word):
        return re.fullmatch(r"[a-zA-Z\-]{2,}", word) is not None

    def extract_keywords(self, chunks):
        keywords = set([])
        texts = [chunk["text"] for chunk in chunks if chunk["text"].strip()]
        for text in texts:
            keywords.update(self._extract_concepts(text))
        keywords = list(keywords)[:self.max_keywords]
        logger.info(f"Extracted keywords: {keywords}")
        return keywords

    def generate_reference_sentences(self, keywords, context):
        references = []
        for kw in keywords:
            prompt = PROMPT.format(n=self.sentences_per_keyword, keyword=kw, context=context )
            try:
                response = self.llm.generate(prompt,REFERENCE_PROMPT)
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
        logger.info(f"Generated {len(references)} {constants.safe_text(context)} reference sentences")
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

class ConceptSet:
    def __init__(self):
        self._unique = set()
        self._freq = defaultdict(int)

    def add(self, word):
        word = self.__normalize_text__(word).lower()
        if word:
            self._unique.add(word)
            self._freq[word] += 1

    def update(self, words):
        for word in words:
            self.add(word)

    def __normalize_text__(self,text):
        text = ftfy.fix_text(text)  # Fix broken Unicode
        text = text.replace("•", "")  # Remove bullet
        return text.strip()

    def __contains__(self, word):
        return word.lower() in self._unique

    def __len__(self):
        return len(self._unique)

    def get_all(self):
        return sorted(self._unique)

    def get_ranked(self, top_n=None, method="combined"):
        def score(word):
            if method == "length":
                return len(word)
            elif method == "frequency":
                return self._freq[word]
            elif method == "combined":
                return len(word) * self._freq[word]
            else:
                raise ValueError(f"Unknown method: {method}")

        ranked = sorted(self._unique, key=score, reverse=True)
        return ranked[:top_n] if top_n else ranked

    def get_frequency(self, word):
        return self._freq.get(word.lower(), 0)

    def summary(self):
        return {word: self._freq[word] for word in sorted(self._unique)}

    def __repr__(self):
        return f"<ConceptSet size={len(self._unique)}>"


def main():
    llm = LocalLLMClient()
    builder = ReferenceBuilder(llm_client=llm, sentences_per_keyword=3)
    keywords = ["Parallelism", "Divide‑and‑Conquer", "Big O Complexity", "Randomized Algorithms", "Dynamic Programming", "Greedy Strategies", "Approximation Ratios", "NP‑Hardness", "Heuristic Search", "Graph Traversal"]
    context = "Computing Algorithmics"
    references = builder.generate_reference_sentences(keywords, context)
    print(references)

if __name__ == "__main__":
    main()