from semantic_filter import SemanticFilter
from chunk_tagger import ChunkTagger
from test_retrieval import RetrievalTester


def main():
    # Load chunks from DB or file
    chunks = load_chunks()

    # Step 1: Filter
    filterer = SemanticFilter()
    reference_texts = ["Bayes theorem", "gradient descent", "ethics in AI"]
    filtered_chunks = filterer.filter_chunks(chunks, reference_texts)

    # Step 2: Tag
    tagger = ChunkTagger()
    tagged_chunks = tagger.tag_chunks(filtered_chunks)

    # Step 3: Test retrieval
    tester = RetrievalTester()
    tester.test_queries(["What is Bayes theorem?", "Explain gradient descent"])


if __name__ == "__main__":
    main()