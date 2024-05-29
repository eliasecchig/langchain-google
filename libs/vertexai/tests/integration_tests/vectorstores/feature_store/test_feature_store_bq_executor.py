"""
Test Vertex Feature Store Vector Search with BQ Vector Search executor.
"""

import os
import uuid

import pytest

from langchain_google_vertexai.vectorstores.feature_store.bq_vectorstore import (
    BigQueryExecutor,
)
from langchain_google_vertexai.vectorstores.feature_store.feature_store import (
    FeatureStore,
)

TEST_TABLE_NAME = "langchain_test_table"


@pytest.fixture(scope="class")
def store_bq_executor(request: pytest.FixtureRequest) -> FeatureStore:
    """BigQueryVectorStore tests context.

    In order to run this test, you define PROJECT_ID environment variable
    with GCP project id.

    Example:
    export PROJECT_ID=...
    """
    from google.cloud import bigquery

    from langchain_google_vertexai import VertexAIEmbeddings

    embedding_model = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest",
        project=os.environ.get("PROJECT_ID", None),
    )
    TestFeatureStore_bq_executor.store_bq_executor = FeatureStore(
        project_id=os.environ.get("PROJECT_ID", None),  # type: ignore[arg-type]
        embedding=embedding_model,
        location="us-central1",
        dataset_name=TestFeatureStore_bq_executor.dataset_name,
        table_name=TEST_TABLE_NAME,
        executor=BigQueryExecutor(),
    )
    TestFeatureStore_bq_executor.store_bq_executor.add_texts(
        TestFeatureStore_bq_executor.texts,
        TestFeatureStore_bq_executor.metadatas,
    )

    def teardown() -> None:
        bigquery.Client(location="us-central1").delete_dataset(
            TestFeatureStore_bq_executor.dataset_name,
            delete_contents=True,
            not_found_ok=True,
        )

    request.addfinalizer(teardown)
    return TestFeatureStore_bq_executor.store_bq_executor


class TestFeatureStore_bq_executor:
    """BigQueryVectorStore tests class."""

    dataset_name = uuid.uuid4().hex
    store_bq_executor: FeatureStore
    texts = ["apple", "ice cream", "Saturn", "candy", "banana"]
    metadatas = [
        {
            "kind": "fruit",
        },
        {
            "kind": "treat",
        },
        {
            "kind": "planet",
        },
        {
            "kind": "treat",
        },
        {
            "kind": "fruit",
        },
    ]

    @pytest.mark.extended
    def test_semantic_search(self, store_bq_executor: FeatureStore) -> None:
        """Test on semantic similarity."""
        docs = store_bq_executor.similarity_search("food", k=4)
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_semantic_search_filter_fruits(
        self, store_bq_executor: FeatureStore
    ) -> None:
        """Test on semantic similarity with metadata filter."""
        docs = store_bq_executor.similarity_search("food", filter={"kind": "fruit"})
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_get_doc_by_filter(self, store_bq_executor: FeatureStore) -> None:
        """Test on document retrieval with metadata filter."""
        docs = store_bq_executor.get_documents(filter={"kind": "fruit"})
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_add_texts_with_embeddings(self, store_bq_executor: FeatureStore) -> None:
        """Test adding texts with pre-computed embeddings."""
        new_texts = ["chocolate", "mars"]
        new_metadatas = [{"kind": "treat"}, {"kind": "planet"}]
        new_embeddings = store_bq_executor.embedding.embed_documents(new_texts)
        ids = store_bq_executor.add_texts_with_embeddings(
            new_texts, new_embeddings, new_metadatas
        )
        assert len(ids) == 2  # Ensure we got IDs back
        # Verify the documents were added correctly
        retrieved_docs = store_bq_executor.get_documents(ids)
        assert retrieved_docs[0].page_content == "chocolate"
        assert retrieved_docs[1].page_content == "mars"
        assert retrieved_docs[0].metadata["kind"] == "treat"
        assert retrieved_docs[1].metadata["kind"] == "planet"

    @pytest.mark.extended
    def test_get_documents_by_ids(self, store_bq_executor: FeatureStore) -> None:
        """Test retrieving documents by their IDs."""
        # Get the first two documents
        first_two_docs = store_bq_executor.get_documents()[:2]
        ids_to_retrieve = [doc.metadata["__id"] for doc in first_two_docs]
        # Retrieve them by their IDs
        retrieved_docs = store_bq_executor.get_documents(ids_to_retrieve)
        assert len(retrieved_docs) == 2
        # Check that the content and metadata match
        for orig_doc, retrieved_doc in zip(first_two_docs, retrieved_docs):
            assert orig_doc.page_content == retrieved_doc.page_content
            assert orig_doc.metadata == retrieved_doc.metadata

    @pytest.mark.extended
    def test_delete_documents(self, store_bq_executor: FeatureStore) -> None:
        """Test deleting documents by their IDs."""
        doc_to_delete = store_bq_executor.get_documents()[0]
        id_to_delete = doc_to_delete.metadata["__id"]
        # Delete the document
        delete_result = store_bq_executor.delete([id_to_delete])
        assert delete_result is True  # Deletion should succeed
        # Try to retrieve the deleted document

        result = store_bq_executor.get_documents([id_to_delete])
        assert result == []

    @pytest.mark.extended
    def test_batch_search(self, store_bq_executor: FeatureStore) -> None:
        """Test batch search with queries and embeddings."""
        # Batch search with queries
        query_results = store_bq_executor.batch_search(queries=["apple", "treat"])
        assert len(query_results) == 2  # 2 queries
        assert all(
            len(result) > 0 for result in query_results
        )  # Results for each query

        # Batch search with embeddings
        embeddings = store_bq_executor.embedding.embed_documents(["apple", "treat"])
        embedding_results = store_bq_executor.batch_search(embeddings=embeddings)
        assert len(embedding_results) == 2  # 2 embeddings
        assert all(len(result) > 0 for result in embedding_results)
