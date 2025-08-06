import constants
from chromadb import Client
from sentence_transformers import SentenceTransformer


class RetrievalTester:
    def __init__(self, collection_name="Unknown", model_name=constants.MODEL_EMBED_NAME):
        self.client = Client()
        self.collection = self.client.get_collection(collection_name)
        self.model = SentenceTransformer(model_name)

    def query(self, text, top_k=5):
        embedding = self.model.encode(text).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        return results["documents"][0]

    def test_queries(self, queries):
        for q in queries:
            print(f"\n🔍 Query: {q}")
            docs = self.query(q)
            for i, doc in enumerate(docs):
                print(f"{i+1}. {doc[:100]}...")