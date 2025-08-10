from chromadb import PersistentClient
from flask import Flask, request, render_template

import re
import constants
from ingestion.query_retrieval import RetrievalQuery

class QueryApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.processor = QueryProcessor()

        @self.app.route("/", methods=["GET"])
        def index():
            return render_template( "index.html", collections=get_collections() )

        @self.app.route("/query", methods=["POST"])
        def query():
            user_query = request.form["query"]
            selected_collection = request.form["collection"]
            result_html = self.processor.process(user_query, selected_collection)
            collections = get_collections()
            return render_template("index.html", result=result_html, collections=collections)

        def get_collections():
            client = PersistentClient(path=constants.DATA_CHROMA_DB_DIR)
            return [c.name for c in client.list_collections()]

    def run(self):
        self.app.run(debug=True)

class QueryProcessor:
    def __init__(self):
        self.formatter = ResultFormatter()

    def process(self, query: str, collection_name: str) -> str:
        retriever = RetrievalQuery(collection_name=collection_name)
        results = retriever.query(query)
        return self.formatter.format(results, query)


class ResultFormatter:
    def format(self, chunks: list, query: str) -> str:
        summary = self.summarize(chunks, query)
        references = self.generate_references(chunks)
        quotes = self.tag_quotes(chunks)
        return f"""
        <div class='summary'>{summary}</div>
        <div class='quotes'>{quotes}</div>
        <div class='references'>{references}</div>
        """

    def summarize(self, chunks, query):
        # Placeholder: use LLM or rule-based summarizer
        return f"<p>Summary of results for: <strong>{query}</strong></p>"

    def generate_references(self, chunks):
        refs = []
        for i, chunk in enumerate(chunks, 1):
            doc = chunk.get("document_name", "Unknown")
            doc = self.extract_relative_path(doc, "static") or doc
            page = chunk.get("page_number", "N/A")
            refs.append( f"<li>[{i}] <a href='/static/{doc}#page={page}' target='_blank'>{doc}, page {page}</a></li>" )

        return "<ul>" + "".join(refs) + "</ul>"

    def tag_quotes(self, chunks):
        quotes = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            quotes.append(f"<p>{text} <sup>[{i}]</sup></p>")
        return "".join(quotes)

    def extract_relative_path(self, full_path, folder_name):
        """
        Extracts the part of the path after the specified folder name.

        Args:
            full_path (str): The full file path.
            folder_name (str): The folder name to anchor the split.

        Returns:
            str: The relative path after the folder, or None if not found.
        """
        pattern = rf"{re.escape(folder_name)}[\\/](.+)"
        match = re.search(pattern, full_path)
        return match.group(1) if match else None


if __name__ == "__main__":
    QueryApp().run()
