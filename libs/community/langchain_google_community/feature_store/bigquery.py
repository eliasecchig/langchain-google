from datetime import datetime, timedelta
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Type, Union, Literal
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import proto  # type: ignore[import-untyped]
import vertexai  # type: ignore[import-untyped]
from google.api_core.exceptions import (
    ClientError,
)
from google.cloud import bigquery
from google.cloud.aiplatform import base, telemetry

from vertexai.resources.preview import (  # type: ignore[import-untyped]
    AlgorithmConfig,
    DistanceMeasureType,
    FeatureOnlineStore,
    FeatureView,
    FeatureViewBigQuerySource,
)
from vertexai.resources.preview.feature_store import (  # type: ignore[import-untyped]
    utils,
)

from langchain_google_community.feature_store._base import BaseBigQueryVectorStore




_vector_table_lock = Lock()  # process-wide BigQueryVectorSearch table lock

logger = base.Logger(__name__)
# Constants for index creation
MIN_INDEX_ROWS = 5
INDEX_CHECK_INTERVAL = timedelta(seconds=60)
USER_AGENT_PREFIX = "FeatureStore"


class BigQueryVectorStore(BaseBigQueryVectorStore):
    distance_type: Literal["COSINE", "EUCLIDEAN"] = "EUCLIDEAN"
    _creating_index: bool = False
    _have_index: bool = False
    _last_index_check: datetime = datetime.min

    def model_post_init(self, __context: Any) -> None:
        # Initialize attributes after model creation
        super().model_post_init(__context)
        self._creating_index = False
        self._have_index = False
        self._last_index_check = datetime.min
        logger.info(
            "BigQueryVectorStore initialized with BigQuery VectorSearch. \n"
            "Optional online serving available via .get_vertex_fs_vector_store() method."
        )

    def sync(self):
        """Sync the data from the Big Query source into the source"""
        self._initialize_bq_vector_index()

    def get_documents(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
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

        if ids and len(ids) > 0:
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("ids", "STRING", ids),
                ]
            )
            id_expr = f"{self.doc_id_field} IN UNNEST(@ids)"
        else:
            job_config = None
            id_expr = "TRUE"
        if filter:
            filter_expressions = []
            for column, value in filter.items():
                filter_expressions.append(f"{column} = '{value}'")
            filter_expression_str = " AND ".join(filter_expressions)
            where_filter_expr = f" AND ({filter_expression_str})"
        else:
            where_filter_expr = ""

        job = self._bq_client.query(  # type: ignore[union-attr]
            f"""
                    SELECT * FROM `{self._full_table_id}` WHERE {id_expr}
                    {where_filter_expr}
                    """,
            job_config=job_config,
        )
        docs: List[Document] = []
        for row in job:
            metadata = {}
            for field in row.keys():
                if field not in [
                    self.text_embedding_field,
                    self.content_field,
                ]:
                    metadata[field] = row[field]
            metadata["__id"] = row[self.doc_id_field]
            doc = Document(
                page_content=row[self.content_field], metadata=metadata
            )
            docs.append(doc)
        return docs

    def _initialize_bq_vector_index(self) -> Any:
        """
        A vector index in BigQuery table enables efficient
        approximate vector search.
        """
        if self._have_index or self._creating_index:
            return

        table = self._bq_client.get_table(self._full_table_id)  # type: ignore[union-attr]
        if (table.num_rows or 0) < MIN_INDEX_ROWS:
            logger.debug("Not enough rows to create a vector index.")
            return

        if datetime.utcnow() - self._last_index_check < INDEX_CHECK_INTERVAL:
            return

        with _vector_table_lock:
            self._last_index_check = datetime.utcnow()
            # Check if index exists, create if necessary
            check_query = (
                f"SELECT 1 FROM `{self.project_id}."
                f"{self.dataset_name}"
                ".INFORMATION_SCHEMA.VECTOR_INDEXES` WHERE"
                f" table_name = '{self.table_name}'"
            )
            job = self._bq_client.query(  # type: ignore[union-attr]
                check_query, api_method=bigquery.enums.QueryApiMethod.QUERY
            )
            if job.result().total_rows == 0:
                # Need to create an index. Make it in a separate thread.
                self._create_bq_index_in_background()
            else:
                logger.debug("Vector index already exists.")
                self._have_index = True

    def _create_bq_index_in_background(self):
        if self._have_index or self._creating_index:
            return

        self._creating_index = True
        logger.debug("Trying to create a vector index.")
        Thread(target=self._create_bq_index, daemon=True).start()

    def _create_bq_index(self):
        table = self._bq_client.get_table(self._full_table_id)  # type: ignore[union-attr]
        if (table.num_rows or 0) < MIN_INDEX_ROWS:
            return

        index_name = f"{self.table_name}_langchain_index"
        try:
            sql = f"""
                CREATE VECTOR INDEX IF NOT EXISTS
                `{index_name}`
                ON `{self._full_table_id}`
                ({self.text_embedding_field})
                OPTIONS(distance_type="{self.distance_type}", index_type="IVF")
            """
            self._bq_client.query(sql).result()  # type: ignore[union-attr]
            self._have_index = True
        except ClientError as ex:
            logger.debug("Vector index creation failed (%s).", ex.args[0])
        finally:
            self._creating_index = False

    def _similarity_search_by_vectors_with_scores_and_embeddings(
        self,
        embeddings: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        batch_size: Union[int, None] = 100,
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

        final_results = []

        for start in range(0, len(embeddings), batch_size):  # type: ignore[arg-type]
            end = start + batch_size  # type: ignore[operator]
            embs_batch = embeddings[start:end]
            final_results.extend(
                self._search_embeddings(embeddings=embs_batch, filter=filter, k=k)
            )
        if len(final_results) == 0:
            return [[]]
        documents = []
        fields = [
            x
            for x in final_results[0].keys()
            if x
            not in [
                self.text_embedding_field,
                self.content_field,
            ]
        ]
        for result in final_results:
            metadata = {}
            for field in fields:
                metadata[field] = result[field]
            documents.append(
                [
                    Document(
                        page_content=result[self.content_field],
                        metadata=metadata,
                    ),
                    metadata["score"],
                    result[self.text_embedding_field],
                ]
            )
        results_chunks = [
            documents[i * k : (i + 1) * k] for i in range(len(embeddings))
        ]
        return results_chunks

    def _search_embeddings(
        self, embeddings, filter: Optional[Dict[str, Any]] = None, k=5
    ):
        if filter:
            filter_expressions = []
            for column, value in filter.items():
                if self.table_schema[column] in ["INTEGER", "FLOAT"]:  # type: ignore[index]
                    filter_expressions.append(f"base.{column} = {value}")
                else:
                    filter_expressions.append(f"base.{column} = '{value}'")
            where_filter_expr = " AND ".join(filter_expressions)
        else:
            where_filter_expr = "TRUE"

        embeddings_query = "with embeddings as (\n"
        for i, emb in enumerate(embeddings):
            embeddings_query += (
                f"SELECT {i} as row_num, @emb_{i} AS text_embedding"
                if i == 0
                else f"\nUNION ALL\nSELECT {i} as row_num, @emb_{i} AS text_embedding"
            )
        embeddings_query += "\n)\n"

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter(f"emb_{i}", "FLOAT64", emb)
                for i, emb in enumerate(embeddings)
            ],
            use_query_cache=True,
            priority=bigquery.QueryPriority.INTERACTIVE,
        )
        full_query = (
            embeddings_query
            + f"""
        SELECT
            base.*,
            query.row_num,
            distance AS score
        FROM VECTOR_SEARCH(
            TABLE `{self._full_table_id}`,
            "text_embedding",
            (SELECT row_num, {self.text_embedding_field} from embeddings),
            distance_type => "{self.distance_type}",
            top_k => {k}
        )
        WHERE {where_filter_expr}
        ORDER BY row_num, score
        """
        )
        results = self._bq_client.query(  # type: ignore[union-attr]
            full_query,
            job_config=job_config,
            api_method=bigquery.enums.QueryApiMethod.QUERY,
        )
        return list(results)

    @classmethod
    def from_texts(
        cls: Type["BigQueryVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "BigQueryVectorStore":
        """Return VectorStore initialized from input texts

        Args:
            texts: List of strings to add to the vectorstore.
            embedding: An embedding model instance for text to vector transformations.
            metadatas: Optional list of metadata records associated with the texts.
                (ie [{"url": "www.myurl1.com", "title": "title1"},
                {"url": "www.myurl2.com", "title": "title2"}])
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        vs_obj = BigQueryVectorStore(embedding=embedding, **kwargs)
        vs_obj.add_texts(texts, metadatas)
        return vs_obj

    def get_vertex_fs_vector_store(
            self,
            **kwargs
    ) -> BaseBigQueryVectorStore:
        from langchain_google_community.feature_store.featurestore import \
            VertexFSVectorStore

        base_params = self.dict(include=BaseBigQueryVectorStore.__fields__.keys())
        all_params = {**base_params, **kwargs}
        fs_obj = VertexFSVectorStore(**all_params)
        return fs_obj