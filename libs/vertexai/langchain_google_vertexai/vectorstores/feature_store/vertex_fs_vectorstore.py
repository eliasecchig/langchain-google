from __future__ import annotations

import time
from datetime import datetime, timedelta
from subprocess import TimeoutExpired
from threading import Lock, Thread
from typing import Any, Dict, List, Literal, MutableSequence, Optional, Union

import numpy as np
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

from langchain_google_vertexai._utils import get_client_info, get_user_agent
from langchain_google_vertexai.vectorstores.feature_store.utils import (
    EnvConfig,
    cast_proto_type,
    doc_match_filter,
)

_vector_table_lock = Lock()  # process-wide BigQueryVectorSearch table lock

logger = base.Logger(__name__)
# Constants for index creation
MIN_INDEX_ROWS = 5
INDEX_CHECK_INTERVAL = timedelta(seconds=60)
USER_AGENT_PREFIX = "FeatureStore"

from __future__ import annotations

import asyncio
import uuid
from functools import partial
from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Type, Union

from google.cloud import bigquery
from google.cloud.aiplatform import base
from google.cloud.exceptions import NotFound
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, validate_call
from langchain_google_vertexai.vectorstores.feature_store.utils import (
    validate_column_in_bq_schema,
)

logger = base.Logger(__name__)


# class FeatureStore():
#     """Google Cloud Feature Store vector store.
#     The FeatureStore aims to facilitate similarity search using different
#         methodologies on Google Cloud including Big Query, Feature Store and a
#         local bruteforce search engine.
#     Big Query is the data source of truth and also the default search
#         methodology (or executor).
#     When lower latency is required, it is possible to move to the feature store
#         executor with one line: my_fs.set_executor({"type": "feature_online_store"}).
#     The data can be synced from BQ to FS using my_fs.sync().
#     Optionally a cron schedule can be passed for automatic sync of BQ data
#     to fs: my_fs.set_executor({
#         "type": "feature_online_store", "cron_schedule": "TZ=Europe/Rome 00 00 01 5 *"
#         })
#
#     Attributes:
#         embedding (Any): An embedding model instance for text to vector transformations.
#         project_id (str): Your Google Cloud Project ID.
#         dataset_name (str): Name of the dataset within BigQuery.
#         table_name (str): Name of the table within the dataset.
#         location (str): Location of your BigQuery dataset (e.g., "europe-west2").
#         executor (Union[BigQueryExecutor, BruteForceExecutor,
#             FeatureOnlineStoreExecutor]): The executor to use for search
#             (defaults to BigQueryExecutor).
#         content_field (str): The field name in the Feature Store that stores the
#             text content.
#         text_embedding_field (str): The field name in the Feature Store that stores
#             the text embeddings.
#         doc_id_field (str): The field name in the Feature Store that stores the
#             document IDs.
#         credentials (Optional[Any]): Optional credentials for Google Cloud
#             authentication.
#
#     To use, you need the following packages installed:
#         google-cloud-bigquery
#     """

class BaseBigQueryStorageVectorStore(VectorStore, BaseModel):
    embedding: Any
    project_id: str
    dataset_name: str
    table_name: str
    location: str
    content_field: str = "content"
    text_embedding_field: str = "text_embedding"
    doc_id_field: str = "doc_id"
    credentials: Optional[Any] = None
    _extra_fields: Union[Dict[str, str], None] = None
    _table_schema: Any = None

    def sync(self):
        raise NotImplementedError()

    def similarity_search_by_vectors_with_scores_and_embeddings(
        self,
        embeddings: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        batch_size: Union[int, None] = None,
    ) -> list[list[list[Any]]]:
        raise NotImplementedError()

    def model_post_init(self, __context):
        """Constructor for FeatureStore."""
        try:
            import pandas as pd  # type: ignore[import-untyped]

            find_spec("pyarrow")
            find_spec("db_types")
            find_spec("langchain_community")
            self._pd = pd
        except ModuleNotFoundError as e:
            logger.error(e)
            raise ImportError(
                "Please, install feature store dependency group: "
                "`pip install langchain-google-vertexai[featurestore]`"
            )
        self._bq_client = bigquery.Client(
            project=self.project_id,
            location=self.location,
            credentials=self.credentials,
        )
        self._embedding_dimension = len(self.embedding.embed_query("test"))
        self._full_table_id = (
            f"{self.project_id}." f"{self.dataset_name}." f"{self.table_name}"
        )
        self._initialize_bq_table()
        self._validate_bq_table()
        logger.info(
            f"BigQuery table {self._full_table_id} "
            f"initialized/validated as persistent storage. "
            f"Access via BigQuery console:\n "
            f"https://console.cloud.google.com/bigquery?project={self.project_id}"
            f"&ws=!1m5!1m4!4m3!1s{self.project_id}!2s{self.dataset_name}!3s{self.table_name}"
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    @property
    def full_table_id(self) -> str:
        return self._full_table_id

    def _validate_bq_table(self):
        table_ref = bigquery.TableReference.from_string(self._full_table_id)

        try:
            table = self._bq_client.get_table(
                self.full_table_id
            )  # Attempt to retrieve the table information
        except NotFound:
            logger.debug(
                f"Couldn't find table {self.full_table_id}. "
                f"Table will be created once documents are added"
            )
            return

        table = self._bq_client.get_table(table_ref)
        schema = table.schema.copy()
        if schema:  ## Check if table has a schema
            self._table_schema = {field.name: field.field_type for field in schema}
            columns = {c.name: c for c in schema}
            validate_column_in_bq_schema(
                column_name=self.doc_id_field,
                columns=columns,
                expected_types=["STRING"],
                expected_modes=["NULLABLE", "REQUIRED"],
            )
            validate_column_in_bq_schema(
                column_name=self.content_field,
                columns=columns,
                expected_types=["STRING"],
                expected_modes=["NULLABLE", "REQUIRED"],
            )
            validate_column_in_bq_schema(
                column_name=self.text_embedding_field,
                columns=columns,
                expected_types=["FLOAT", "FLOAT64"],
                expected_modes=["REPEATED"],
            )
            if self._extra_fields is None:
                extra_fields = {}
                for column in schema:
                    if column.name not in [
                        self.doc_id_field,
                        self.content_field,
                        self.text_embedding_field,
                    ]:
                        # Check for unsupported REPEATED mode
                        if column.mode == "REPEATED":
                            raise ValueError(
                                f"Column '{column.name}' is REPEATED. "
                                f"REPEATED fields are not supported in this context."
                            )
                        extra_fields[column.name] = column.field_type
                self._extra_fields = extra_fields
                # if self:
                #     self.extra_fields = extra_fields
                #     self.table_schema = self._table_schema
            else:
                for field, type in self._extra_fields.items():
                    validate_column_in_bq_schema(
                        column_name=field,
                        columns=columns,
                        expected_types=[type],
                        expected_modes=["NULLABLE", "REQUIRED"],
                    )
            logger.debug(f"Table {self.full_table_id} validated")
        return table_ref

    def _initialize_bq_table(self) -> Any:
        """Validates or creates the BigQuery table."""
        self._bq_client.create_dataset(dataset=self.dataset_name, exists_ok=True)
        table_ref = bigquery.TableReference.from_string(self._full_table_id)
        self._bq_client.create_table(table_ref, exists_ok=True)
        return table_ref

    def add_texts(  # type: ignore[override]
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: List of strings to add to the vectorstore.
            metadatas: Optional list of metadata records associated with the texts.
                (ie [{"url": "www.myurl1.com", "title": "title1"},
                {"url": "www.myurl2.com", "title": "title2"}])

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embs = self.embedding.embed_documents(texts)
        return self.add_texts_with_embeddings(
            texts=texts, embs=embs, metadatas=metadatas, **kwargs
        )

    def add_texts_with_embeddings(
        self,
        texts: List[str],
        embs: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add precomputed embeddings and relative texts / metadatas to the vectorstore.

        Args:
            ids: List of unique ids in string format
            texts: List of strings to add to the vectorstore.
            embs: List of lists of floats with text embeddings for texts.
            metadatas: Optional list of metadata records associated with the texts.
                (ie [{"url": "www.myurl1.com", "title": "title1"},
                {"url": "www.myurl2.com", "title": "title2"}])
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        ids = [uuid.uuid4().hex for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        values_dict: List[Dict[str, List[Any]]] = []
        for idx, text, emb, metadata_dict in zip(ids, texts, embs, metadatas):
            record = {
                self.doc_id_field: idx,
                self.content_field: text,
                self.text_embedding_field: emb,
            }
            record.update(metadata_dict)
            values_dict.append(record)  # type: ignore[arg-type]

        table = self._bq_client.get_table(
            self.full_table_id
        )  # Attempt to retrieve the table information
        df = self._pd.DataFrame(values_dict)
        job = self._bq_client.load_table_from_dataframe(df, table)
        job.result()
        self._validate_bq_table()
        logger.debug(f"stored {len(ids)} records in BQ")
        self.sync()
        return ids

    def get_documents(
        self,
        ids: Optional[List[str]] = None,
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
        raise NotImplementedError

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by record IDs

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if not ids or len(ids) == 0:
            return True

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", ids)],
        )
        self._bq_client.query(
            f"""
                    DELETE FROM `{self.full_table_id}` WHERE {self.doc_id_field}
                    IN UNNEST(@ids)
                    """,
            job_config=job_config,
        ).result()
        self.sync()
        return True

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.delete, **kwargs), ids
        )

    def similarity_search_by_vectors(
        self,
        embeddings: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        with_scores: bool = False,
        with_embeddings: bool = False,
        **kwargs: Any,
    ) -> List[List[Document]]:
        """Core similarity search function. Handles a list of embedding vectors,
            optionally returning scores and embeddings.

        Args:
            embeddings: A list of embedding vectors, where each vector is a list of
                floats.
            filter: (Optional) A dictionary specifying filtering criteria for the
                documents.
                Ie. {"title": "mytitle"}
            k: (Optional) The number of top-ranking similar documents to return per
                embedding. Defaults to 5.
            with_scores: (Optional) If True, include similarity scores in the result
                for each matched document. Defaults to False.
            with_embeddings: (Optional) If True, include the matched document's
                embedding vector in the result. Defaults to False.
        Returns:
            A list of `k` documents for each embedding in `embeddings`
        """
        results = self.similarity_search_by_vectors_with_scores_and_embeddings(
            embeddings=embeddings, k=k, filter=filter, **kwargs
        )

        # Process results based on options
        for i, query_results in enumerate(results):
            if not with_scores and not with_embeddings:
                # return only docs
                results[i] = [x[0] for x in query_results]
            elif not with_embeddings:
                # return only docs and score
                results[i] = [[x[0], x[1]] for x in query_results]
            elif not with_scores:
                # return only docs and embeddings
                results[i] = [[x[0], x[2]] for x in query_results]

        return results  # type: ignore[return-value]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 5,
        **kwargs,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            filter: (Optional) A dictionary specifying filtering criteria for the
                documents. Ie. {"title": "mytitle"}
            k: (Optional) The number of top-ranking similar documents to return per
                embedding. Defaults to 5.
        Returns:
            Return docs most similar to embedding vector.
        """
        return self.similarity_search_by_vectors(embeddings=[embedding], k=k, **kwargs)[
            0
        ]

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
    ):
        """Return docs most similar to embedding vector with scores.

        Args:
            embedding: Embedding to look up documents similar to.
            filter: (Optional) A dictionary specifying filtering criteria for the
                documents. Ie. {"title": "mytitle"}
            k: (Optional) The number of top-ranking similar documents to return per
                embedding. Defaults to 5.
        Returns:
            Return docs most similar to embedding vector.
        """
        return self.similarity_search_by_vectors(
            embeddings=[embedding], filter=filter, k=k, with_scores=True
        )[0]

    def similarity_search(self, query: str, k: int = 5, **kwargs):
        """Search for top `k` docs most similar to input query.

        Args:
            query: search query to search documents with.
            filter: (Optional) A dictionary specifying filtering criteria for the
                documents. Ie. {"title": "mytitle"}
            k: (Optional) The number of top-ranking similar documents to return per
                embedding. Defaults to 5.
        Returns:
            Return docs most similar to input query.
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vectors(embeddings=[embedding], k=k, **kwargs)[
            0
        ]

    def similarity_search_with_score(
        self, query: str, filter: Optional[Dict[str, Any]] = None, k: int = 5, **kwargs
    ):
        """Search for top `k` docs most similar to input query, returns both docs and
            scores.

        Args:
            query: search query to search documents with.
            filter: (Optional) A dictionary specifying filtering criteria for the
                documents. Ie. {"title": "mytitle"}
            k: (Optional) The number of top-ranking similar documents to return per
                embedding. Defaults to 5.
        Returns:
            Return docs most similar to input query along with scores.
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(
            embedding=embedding, filter=filter, k=k, **kwargs
        )

    def batch_search(
        self,
        embeddings: Optional[List[List[float]]] = None,
        queries: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        with_scores: bool = False,
        with_embeddings: bool = False,
        **kwargs,
    ):
        """Multi-purpose batch search function. Accepts either embeddings or queries
            but not both. Optionally returns similarity scores and/or matched embeddings
        Args:
        embeddings: A list of embeddings to search with. If provided, each
            embedding represents a query vector.
        queries: A list of text queries to search with.  If provided, each
            query represents a query text.
        filter: A dictionary of filters to apply to the search. The keys
            of the dictionary should be field names, and the values should be the
                values to filter on. (e.g., {"category": "news"})
        k: The number of top results to return per query. Defaults to 5.
        with_scores: If True, returns the relevance scores of the results along with
            the documents
        with_embeddings: If True, returns the embeddings of the results along with
            the documents
        """
        if not embeddings and not queries:
            raise ValueError(
                "At least one of 'embeddings' or 'queries' must be provided."
            )

        if embeddings is not None and queries is not None:
            raise ValueError(
                "Only one parameter between 'embeddings' or 'queries' must be provided"
            )

        if queries is not None:
            embeddings = self.embedding.embed_documents(queries)

        if embeddings is not None:
            return self.similarity_search_by_vectors(
                embeddings=embeddings,
                filter=filter,
                k=k,
                with_scores=with_scores,
                with_embeddings=with_embeddings,
                **kwargs,
            )

    @classmethod
    def from_texts(
        cls: Type["FeatureStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "FeatureStore":
        raise NotImplementedError()

    # @validate_call
    # def set_executor(
    #     self,
    #     executor: Union[
    #         BigQueryExecutor, FeatureOnlineStoreExecutor, BruteForceExecutor
    #     ],
    # ):
    #     """Set a different executor to run similarity search.
    # 
    #     Args:
    #         executor: Any of [BigQueryExecutor, FeatureOnlineStoreExecutor,
    #             BruteForceExecutor]
    #             example usage:
    #                 1. my_fs.set_executor({"type": "big_query"})
    #                 2. my_fs.set_executor({"type": "feature_online_store",
    #                     "cron_schedule": "TZ=Europe/Rome 00 00 01 5 *"})
    #                 3. my_fs.set_executor({"type": "brute_force"})
    #     Returns:
    #         None
    #     """
    #     self.executor = executor
    #     if self is not None:
    #         self.executor.set_env_config(env_config=self)


class BigQueryVectorStore(BaseBigQueryStorageVectorStore):
    type: Literal["bigquery"] = "bigquery"
    distance_type: Literal["COSINE", "EUCLIDEAN"] = "EUCLIDEAN"
    _creating_index: bool = False
    _have_index: bool = False
    _last_index_check: datetime = datetime.min

    def model_post_init(self, __context: Any) -> None:
        # Initialize attributes after model creation
        self._creating_index = False
        self._have_index = False
        self._last_index_check = datetime.min

    def sync(self):
        """Sync the data from the Big Query source into the source"""
        self._initialize_bq_vector_index()

    def get_documents(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
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

        job = self.bq_client.query(  # type: ignore[union-attr]
            f"""
                    SELECT * FROM `{self.full_table_id}` WHERE {id_expr}
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

        table = self.bq_client.get_table(self.full_table_id)  # type: ignore[union-attr]
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
            job = self.bq_client.query(  # type: ignore[union-attr]
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
        table = self.bq_client.get_table(self.full_table_id)  # type: ignore[union-attr]
        if (table.num_rows or 0) < MIN_INDEX_ROWS:
            return

        index_name = f"{self.table_name}_langchain_index"
        try:
            sql = f"""
                CREATE VECTOR INDEX IF NOT EXISTS
                `{index_name}`
                ON `{self.full_table_id}`
                ({self.text_embedding_field})
                OPTIONS(distance_type="{self.distance_type}", index_type="IVF")
            """
            self.bq_client.query(sql).result()  # type: ignore[union-attr]
            self._have_index = True
        except ClientError as ex:
            logger.debug("Vector index creation failed (%s).", ex.args[0])
        finally:
            self._creating_index = False

    def similarity_search_by_vectors_with_scores_and_embeddings(
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
            TABLE `{self.full_table_id}`,
            "text_embedding",
            (SELECT row_num, {self.text_embedding_field} from embeddings),
            distance_type => "{self.distance_type}",
            top_k => {k}
        )
        WHERE {where_filter_expr}
        ORDER BY row_num, score
        """
        )
        results = self.bq_client.query(  # type: ignore[union-attr]
            full_query,
            job_config=job_config,
            api_method=bigquery.enums.QueryApiMethod.QUERY,
        )
        return list(results)

    @classmethod
    def from_texts(
        cls: Type["FeatureStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "FeatureStore":
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

#
# class VertexFSVectorStore(BaseBigQueryVectorStore):
#     model_config = ConfigDict(arbitrary_types_allowed=True)
#     type: Literal["feature_online_store"] = "feature_online_store"
#     online_store_name: Union[str, None] = None
#     view_name: Union[str, None] = None
#     online_store_type: Literal["bigtable", "optimized"] = "optimized"
#     cron_schedule: Union[str, None] = None
#     location: Union[str, None] = None
#     min_node_count: int = 1
#     max_node_count: int = 3
#     cpu_utilization_target: int = 50
#     algorithm_config: AlgorithmConfig = utils.TreeAhConfig()
#     filter_columns: Optional[List[str]] = None
#     crowding_column: Optional[str] = None
#     distance_measure_type: Optional[DistanceMeasureType] = (
#         utils.DistanceMeasureType.DOT_PRODUCT_DISTANCE
#     )
#     _env_config: EnvConfig = EnvConfig()
#     _user_agent: str = ""
#
#     def model_post_init(self, __context: Any) -> None:
#         _, self._user_agent = get_user_agent(
#             f"{USER_AGENT_PREFIX}-{type(self).__name__}"
#         )
#
#     def set_env_config(self, env_config: Any):
#         super().set_env_config(env_config)
#         self.init_feature_store()
#
#     def _validate_bq_existing_source(
#         self,
#         project_id_param,
#         dataset_param,
#         table_param,
#     ):
#         bq_uri_split = self._feature_view.gca_resource.big_query_source.uri.split(".")  # type: ignore[union-attr]
#         project_id = bq_uri_split[0].replace("bq://", "")
#         dataset = bq_uri_split[1]
#         table = bq_uri_split[2]
#         try:
#             assert project_id == project_id_param
#             assert dataset == dataset_param
#             assert table == table_param
#         except AssertionError:
#             error_message = (
#                 "The BQ table passed in input is"
#                 f"bq://{project_id_param}.{dataset_param}.{table_param} "
#                 f"while the BQ table linked to the feature view is "
#                 "{self._feature_view.gca_resource.big_query_source.uri}."
#                 "Make sure you are using the same table for the feature "
#                 "view."
#             )
#             raise AssertionError(error_message)
#
#     def init_feature_store(self):
#         self.online_store_name = self.online_store_name or self.dataset_name
#         self.view_name = self.view_name or self.table_name
#         self.location = self.location or self.location
#         vertexai.init(project=self.project_id, location=self.location)
#
#         api_endpoint = f"{self.location}-aiplatform.googleapis.com"
#         self._admin_client = FeatureOnlineStoreAdminServiceClient(
#             client_options={"api_endpoint": api_endpoint},
#             client_info=get_client_info(module=self._user_agent),
#         )
#         self._online_store = self._create_online_store()
#         self._search_client = self._get_search_client()
#         self._feature_view = self._get_feature_view()
#
#     def _get_search_client(self) -> FeatureOnlineStoreServiceClient:
#         gca_resource = self._online_store.gca_resource
#         endpoint = gca_resource.dedicated_serving_endpoint.public_endpoint_domain_name
#         return FeatureOnlineStoreServiceClient(
#             client_options={"api_endpoint": endpoint}
#         )
#
#     def _wait_until_dummy_query_success(self, timeout_seconds: int = 1200):
#         """
#         Waits until a dummy query succeeds, indicating the system is ready.
#         """
#         start_time = time.time()
#
#         while True:
#             elapsed_time = time.time() - start_time
#             if elapsed_time > timeout_seconds:
#                 raise TimeoutExpired(
#                     "Timeout of {} seconds exceeded".format(timeout_seconds),
#                     timeout=timeout_seconds,
#                 )
#             try:
#                 return self._search_embedding(
#                     embedding=[1] * self.embedding_dimension,  # type: ignore[operator]
#                     k=1,
#                 )
#             except ServiceUnavailable:
#                 logger.info(
#                     "DNS certificates are being propagated,"
#                     " waiting for 10 seconds.  "
#                 )
#                 time.sleep(10)
#             except MethodNotImplemented as e:
#                 if e.args and "Received http2 header with status" in e.args[0]:
#                     logger.info(
#                         "DNS certificates are being propagated,"
#                         " waiting for 10 seconds.  "
#                     )
#                     time.sleep(10)
#                 else:
#                     raise
#
#     def sync(self):
#         """Sync the data from the Big Query source into the Executor source"""
#         self._feature_view = self._create_feature_view()
#         self._validate_bq_existing_source(
#             project_id_param=self.project_id,
#             dataset_param=self.dataset_name,
#             table_param=self.table_name,
#         )
#         sync_response = self._admin_client.sync_feature_view(
#             feature_view=(
#                 f"projects/{self.project_id}/"
#                 f"locations/{self.location}"
#                 f"/featureOnlineStores/{self.online_store_name}"
#                 f"/featureViews/{self.view_name}"
#             )
#         )
#         while True:
#             feature_view_sync = self._admin_client.get_feature_view_sync(
#                 name=sync_response.feature_view_sync
#             )
#             if feature_view_sync.run_time.end_time.seconds > 0:
#                 status = (
#                     "Succeed" if feature_view_sync.final_status.code == 0 else "Failed"
#                 )
#                 logger.info(f"Sync {status} for {feature_view_sync.name}.")
#                 break
#             else:
#                 logger.info("Sync ongoing, waiting for 30 seconds.")
#             time.sleep(30)
#
#         self._wait_until_dummy_query_success()
#
#     def similarity_search_by_vectors_with_scores_and_embeddings(
#         self,
#         embeddings: List[List[float]],
#         filter: Optional[Dict[str, Any]] = None,
#         k: int = 5,
#         batch_size: Union[int, None] = None,
#         **kwargs,
#     ) -> list[list[list[Any]]]:
#         """Performs a similarity search using vector embeddings
#
#         This function takes a set of query embeddings and searches for similar documents
#         It returns the top-k matching documents, along with their similarity scores
#         and their corresponding embeddings.
#
#         Args:
#             embeddings: A list of lists, where each inner list represents a
#                 query embedding.
#             filter: (Optional) A dictionary specifying filter criteria for document
#                 on metadata properties, e.g.
#                             {
#                                 "str_property": "foo",
#                                 "int_property": 123
#                             }
#             k: The number of top results to return for each query.
#             batch_size: The size of batches to process embeddings.
#
#         Returns:
#             A list of lists of lists. Each inner list represents the results for a
#                 single query, and contains elements of the form
#                 [Document, score, embedding], where:
#                 - Document: The matching document object.
#                 - score: The similarity score between the query and document.
#                 - embedding: The document's embedding.
#         """
#         output = []
#         for query_embedding in embeddings:
#             documents = []
#             results = self._search_embedding(embedding=query_embedding, k=k, **kwargs)
#
#             for result in results:
#                 metadata, embedding = {}, None
#
#                 for feature in result.entity_key_values.key_values.features:
#                     if feature.name not in [
#                         self.text_embedding_field,
#                         self.content_field,
#                     ]:
#                         dict_values = proto.Message.to_dict(feature.value)
#                         col_type, value = next(iter(dict_values.items()))
#                         value = cast_proto_type(column=col_type, value=value)
#                         metadata[feature.name] = value
#                     if feature.name == self.text_embedding_field:
#                         embedding = feature.value.double_array_value.values
#                     if feature.name == self.content_field:
#                         dict_values = proto.Message.to_dict(feature.value)
#                         content = list(dict_values.values())[0]
#                 if filter is not None and not doc_match_filter(
#                     document=metadata, filter=filter
#                 ):
#                     continue
#                 documents.append(
#                     [
#                         Document(
#                             page_content=content,
#                             metadata=metadata,
#                         ),
#                         result.distance,
#                         embedding,
#                     ]
#                 )
#             output.append(documents)
#         return output
#
#     def get_documents(
#         self,
#         ids: Optional[List[str]],
#         filter: Optional[Dict[str, Any]] = None,
#         **kwargs,
#     ) -> List[Document]:
#         """Search documents by their ids or metadata values.
#         Args:
#             ids: List of ids of documents to retrieve from the vectorstore.
#             filter: Filter on metadata properties, e.g.
#                             {
#                                 "str_property": "foo",
#                                 "int_property": 123
#                             }
#         Returns:
#             List of ids from adding the texts into the vectorstore.
#         """
#         output = []
#         if ids is None:
#             raise ValueError(
#                 "Feature Store executor doesn't support search by filter " "only"
#             )
#         for id in ids:
#             with telemetry.tool_context_manager(self._user_agent):
#                 result = self._feature_view.read(key=[id])  # type: ignore[union-attr]
#                 metadata, content = {}, None
#                 for feature in result.to_dict()["features"]:
#                     if feature["name"] not in [
#                         self.text_embedding_field,
#                         self.content_field,
#                     ]:
#                         metadata[feature["name"]] = list(feature["value"].values())[0]
#                     if feature["name"] == self.content_field:
#                         content = list(feature["value"].values())[0]
#                 if filter is not None and not doc_match_filter(
#                     document=metadata, filter=filter
#                 ):
#                     continue
#                 output.append(
#                     Document(
#                         page_content=str(content),
#                         metadata=metadata,
#                     )
#                 )
#         return output
#
#     def search_neighbors_by_ids(
#         self, ids: List[str], filter: Optional[Dict[str, Any]] = None, **kwargs
#     ) -> List[Document]:
#         """Searches for neighboring entities in a Vertex Feature Store based on
#             their IDs and optional filter on metadata
#
#         Args:
#             ids: A list of string identifiers representing the entities to search for.
#             filter: (Optional) A dictionary specifying filter criteria for document
#                     on metadata properties, e.g.
#                                 {
#                                     "str_property": "foo",
#                                     "int_property": 123
#                                 }
#         """
#         output = []
#         if ids is None:
#             raise ValueError(
#                 "Feature Store executor doesn't support search by filter " "only"
#             )
#         for entity_id in ids:
#             results = self._search_embedding(entity_id=entity_id, **kwargs)
#             for result in results:
#                 metadata, embedding = {}, None
#                 for feature in result.entity_key_values.key_values.features:
#                     if feature.name not in [
#                         self.text_embedding_field,
#                         self.content_field,
#                     ]:
#                         dict_values = proto.Message.to_dict(feature.value)
#                         metadata[feature.name] = list(dict_values.values())[0]
#                     if feature.name == self.text_embedding_field:
#                         embedding = feature.value.double_array_value
#                     if feature.name == self.content_field:
#                         dict_values = proto.Message.to_dict(feature.value)
#                         content = list(dict_values.values())[0]
#                 if filter is not None and not doc_match_filter(
#                     document=metadata, filter=filter
#                 ):
#                     continue
#                 output.append(
#                     [
#                         Document(
#                             page_content=content,
#                             metadata=metadata,
#                         ),
#                         result.distance,
#                         embedding,
#                     ]
#                 )
#
#         return output  # type: ignore[return-value]
#
#     def _search_embedding(
#         self,
#         embedding: Optional[Any] = None,
#         entity_id: Optional[str] = None,
#         k: int = 5,
#         string_filters: Optional[List[NearestNeighborQuery.StringFilter]] = None,
#         per_crowding_attribute_neighbor_count: Optional[int] = None,
#         approximate_neighbor_candidates: Optional[int] = None,
#         leaf_nodes_search_fraction: Optional[float] = None,
#     ) -> MutableSequence[Any]:
#         if embedding:
#             embedding = NearestNeighborQuery.Embedding(value=embedding)
#         query = NearestNeighborQuery(
#             entity_id=entity_id,
#             embedding=embedding,
#             neighbor_count=k,
#             string_filters=string_filters,
#             per_crowding_attribute_neighbor_count=per_crowding_attribute_neighbor_count,
#             parameters={
#                 "approximate_neighbor_candidates": approximate_neighbor_candidates,
#                 "leaf_nodes_search_fraction": leaf_nodes_search_fraction,
#             },
#         )
#         with telemetry.tool_context_manager(self._user_agent):
#             result = self._search_client.search_nearest_entities(
#                 request=feature_online_store_service.SearchNearestEntitiesRequest(
#                     feature_view=self._feature_view.gca_resource.name,  # type: ignore[union-attr]
#                     query=query,
#                     return_full_entity=True,  # returning entities with metadata
#                 )
#             )
#         return result.nearest_neighbors.neighbors
#
#     def _create_online_store(self) -> FeatureOnlineStore:
#         # Search for existing Online store
#         stores_list = FeatureOnlineStore.list(
#             project=self.project_id, location=self.location
#         )
#         for store in stores_list:
#             if store.name == self.online_store_name:
#                 return store
#
#         # Create it otherwise
#         if self.online_store_type == "bigtable":
#             online_store_config = feature_online_store_pb2.FeatureOnlineStore(
#                 bigtable=feature_online_store_pb2.FeatureOnlineStore.Bigtable(
#                     auto_scaling=feature_online_store_pb2.FeatureOnlineStore.Bigtable.AutoScaling(
#                         min_node_count=self.min_node_count,
#                         max_node_count=self.max_node_count,
#                         cpu_utilization_target=self.cpu_utilization_target,
#                     )
#                 ),
#                 embedding_management=feature_online_store_pb2.FeatureOnlineStore.EmbeddingManagement(
#                     enabled=True
#                 ),
#             )
#             create_store_lro = self._admin_client.create_feature_online_store(
#                 parent=f"projects/{self.project_id}/locations/{self.location}",
#                 feature_online_store_id=self.online_store_name,
#                 feature_online_store=online_store_config,
#             )
#             logger.info(create_store_lro.result())
#         elif self.online_store_type == "optimized":
#             online_store_config = feature_online_store_pb2.FeatureOnlineStore(
#                 optimized=feature_online_store_pb2.FeatureOnlineStore.Optimized()
#             )
#             create_store_lro = self._admin_client.create_feature_online_store(
#                 parent=f"projects/{self.project_id}/locations/{self.location}",
#                 feature_online_store_id=self.online_store_name,
#                 feature_online_store=online_store_config,
#             )
#             logger.info(create_store_lro.result())
#             logger.info(create_store_lro.result())
#
#         else:
#             raise ValueError(
#                 f"{self.online_store_type} not allowed. "
#                 f"Accepted values are 'bigtable' or 'optimized'."
#             )
#         stores_list = FeatureOnlineStore.list(
#             project=self.project_id, location=self.location
#         )
#         for store in stores_list:
#             if store.name == self.online_store_name:
#                 return store
#
#     def _create_feature_view(self) -> FeatureView:
#         fv = self._get_feature_view()
#         if fv:
#             return fv
#         else:
#             big_query_source = FeatureViewBigQuerySource(
#                 uri=f"bq://{self.full_table_id}",
#                 entity_id_columns=[self.doc_id_field],
#             )
#             index_config = utils.IndexConfig(
#                 embedding_column=self.text_embedding_field,
#                 crowding_column=self.crowding_column,
#                 filter_columns=self.filter_columns,
#                 dimensions=self.embedding_dimension,
#                 distance_measure_type=self.distance_measure_type,
#                 algorithm_config=self.algorithm_config,
#             )
#             return self._online_store.create_feature_view(
#                 name=self.view_name,
#                 source=big_query_source,
#                 sync_config=self.cron_schedule,
#                 index_config=index_config,
#                 project=self.project_id,
#                 location=self.location,
#             )
#
#     def _get_feature_view(self) -> FeatureView | None:
#         # Search for existing Feature view
#         fv_list = FeatureView.list(
#             feature_online_store_id=self._online_store.gca_resource.name
#         )
#         for fv in fv_list:
#             if fv.name == self.view_name:
#                 return fv
#         return None
#     @classmethod
#     def from_texts(
#         cls: Type["FeatureStore"],
#         texts: List[str],
#         embedding: Embeddings,
#         metadatas: Optional[List[dict]] = None,
#         **kwargs: Any,
#     ) -> "VertexFSVectorStore":
#         """Return VectorStore initialized from input texts
#
#         Args:
#             texts: List of strings to add to the vectorstore.
#             embedding: An embedding model instance for text to vector transformations.
#             metadatas: Optional list of metadata records associated with the texts.
#                 (ie [{"url": "www.myurl1.com", "title": "title1"},
#                 {"url": "www.myurl2.com", "title": "title2"}])
#         Returns:
#             List of ids from adding the texts into the vectorstore.
#         """
#         vs_obj = VertexFSVectorStore(embedding=embedding, **kwargs)
#         vs_obj.add_texts(texts, metadatas)
#         return vs_obj