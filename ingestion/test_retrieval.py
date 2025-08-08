import constants
import mysql.connector
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


class RetrievalTester:
    def __init__(self, collection_name="Unknown", model_name=constants.MODEL_EMBED_NAME):
        self.client = PersistentClient(path=constants.DATA_CHROMA_DB_DIR)
        self.collection = self.client.get_collection(collection_name)
        self.model = SentenceTransformer(model_name)
        self.db_conn = self._connect_mysql()

    def _connect_mysql(self):
        return mysql.connector.connect(
            host=constants.MYSQL_HOST,
            user=constants.MYSQL_USER,
            password=constants.MYSQL_PASSWORD,
            database=constants.MYSQL_SCHEMA
        )

    def _get_metadata(self, doc_id):
        cursor = self.db_conn.cursor(dictionary=True)
        query = "SELECT document_name, page_number FROM document_metadata WHERE id = %s"
        cursor.execute(query, (doc_id,))
        result = cursor.fetchone()
        cursor.close()
        return result or {"document_name": "Unknown", "page_number": -1}

    def query(self, text, top_k=5):
        embedding = self.model.encode(text).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        documents = results["documents"][0]
        ids = results["ids"][0]
        scores = results.get("distances", [])[0] if "distances" in results else [None] * len(documents)

        structured_results = []
        for doc, doc_id, score in zip(documents, ids, scores):
            metadata = self._get_metadata(doc_id)
            structured_results.append({
                "text": doc,
                "document_id": doc_id,
                "document_name": metadata["document_name"],
                "page_number": metadata["page_number"],
                "score": score
            })

        return structured_results

    def test_queries(self, queries):
        all_results = []
        for q in queries:
            print(f"\n🔍 Query: {q}")
            results = self.query(q)
            for i, res in enumerate(results):
                print(f"{i+1}. {res['text'][:300]}... (📄 {res['document_name']}, 📄 Page {res['page_number']})")
            all_results.append({
                "query": q,
                "results": results
            })
        return all_results


def main():
    tester = RetrievalTester(collection_name="159.341")
    results = tester.test_queries([
        "What are processes",
        "What are the benefits of multithreading",
        "What is Amdahl's Law?",
        "What are POSIX threads?"
    ])
    print("\n✅ Done!")
    # Optional: export to JSON file
    # import json
    # with open("retrieval_results.json", "w") as f:
    #     json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
