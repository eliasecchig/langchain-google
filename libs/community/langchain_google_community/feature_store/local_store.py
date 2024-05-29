from typing import Any, Dict, List, Optional, Union
from langchain_core.documents import Document
from google.cloud import bigquery
from google.cloud.aiplatform import base
from langchain_google_community.feature_store._base import BaseBigQueryVectorStore
from langchain_google_community.feature_store.utils import doc_match_filter
import numpy as np

logger = base.Logger(__name__)



class BigQueryInMemoryVectorStore(BaseBigQueryVectorStore):
    _df: Any = None
    _vectors: Any = None
    _vectors_transpose: Any = None
    _df_records: Any = None

    def sync(self):
        """Sync the data from the Big Query source into the Executor source"""
        self._df = self._query_table_to_df()
        self._vectors = np.array(
            self._df[self.text_embedding_field].tolist()
        )
        self._vectors_transpose = self._vectors.T
        self._df_records = self._df.drop(
            columns=[self.text_embedding_field]
        ).to_dict("records")

    def _similarity_search_by_vectors_with_scores_and_embeddings(
        self,
        embeddings: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        batch_size: Union[int, None] = None,
    ) -> list[list[list[Any]]]:
        """Performs a similarity search using vector embeddings

        This function takes a set of query embeddings and searches for similar documents
        It returns the top-k matching documents, along with their similarity scores
        and their corresponding embeddings.

        Args:
            embeddings: A list of lists, where each inner list represents a
                query embedding.
            filter: (Optional) A dictionary specifying filter criteria for document
                on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
            k: The number of top results to return for each query.
            batch_size: The size of batches to process embeddings.

        Returns:
            A list of lists of lists. Each inner list represents the results for a
                single query, and contains elements of the form
                [Document, score, embedding], where:
                - Document: The matching document object.
                - score: The similarity score between the query and document.
                - embedding: The document's embedding.
        """

        if self._df is None:
            raise ValueError(
                "Brute force executor was correctly initialized but not "
                "synced yet. Please run FeatureStore - sync() method"
                " sync the index in memory"
            )
        scores = embeddings @ self._vectors_transpose
        sorted_indices = np.argsort(-scores)[:, :k]

        results = [np.array(self._df_records)[x] for x in sorted_indices]
        top_scores = scores[np.arange(len(embeddings))[:, np.newaxis], sorted_indices]
        top_embeddings = self._vectors[sorted_indices]

        documents = []
        for query_results, query_scores, embeddings_results in zip(
            results, top_scores, top_embeddings
        ):
            query_docs = []
            for doc, doc_score, embedding in zip(
                query_results, query_scores, embeddings_results
            ):
                if filter is not None and not doc_match_filter(
                    document=doc, filter=filter
                ):
                    continue
                query_docs.append(
                    [
                        Document(
                            page_content=doc[self.content_field],
                            metadata=doc,
                        ),
                        doc_score,
                        embedding,
                    ]
                )

            documents.append(query_docs)
        return documents

    def get_documents(
        self,
        ids: Optional[List[str]],
        filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """Search documents by their ids or metadata values.

        Args:
            ids: List of ids of documents to retrieve from the vectorstore.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        if self._df is None:
            raise ValueError(
                "Brute force executor was correctly initialized but not "
                "synced yet. Please run FeatureStore - sync() method"
                " sync the index in memory"
            )
        output = []
        df = self._df
        if ids:
            results = df.loc[df[self.doc_id_field].isin(ids)]
        else:
            results = df
        for i, row in results.iterrows():
            metadata = {}
            for field in row.keys():
                if field not in [
                    self.text_embedding_field,
                    self.content_field,
                ]:
                    metadata[field] = row[field]
            metadata["__id"] = row[self.doc_id_field]
            if filter is not None and not doc_match_filter(
                document=metadata, filter=filter
            ):
                continue
            doc = Document(
                page_content=row[self.content_field], metadata=metadata
            )
            output.append(doc)
        return output

    def _query_table_to_df(self):
        client = self._bq_client
        extra_fields = self._extra_fields
        if extra_fields is None:
            extra_fields = {}
        metadata_fields = list(extra_fields.keys())
        metadata_fields_str = ", ".join(metadata_fields)

        table = (
            f"{self.project_id}.{self.dataset_name}"
            f".{self.table_name}"
        )
        fields = (
            f"{self.doc_id_field}, {self.content_field}, "
            f"{self.text_embedding_field}, {metadata_fields_str}"
        )
        query = f"""
        SELECT {fields}
        FROM {table}
        """
        # Create a query job to read the data
        logger.info(f"Reading data from {table}. It might take a few minutes...")
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True, priority=bigquery.QueryPriority.INTERACTIVE
        )
        query_job = client.query(query, job_config=job_config)  # type: ignore[union-attr]
        return query_job.to_dataframe()