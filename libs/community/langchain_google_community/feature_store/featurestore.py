from __future__ import annotations

import time
from datetime import timedelta
from subprocess import TimeoutExpired
from typing import Literal, MutableSequence

import proto  # type: ignore[import-untyped]
import vertexai  # type: ignore[import-untyped]

from google.cloud.aiplatform import telemetry
from pydantic import ConfigDict

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

from langchain_google_community._utils import get_client_info, get_user_agent
from langchain_google_community.feature_store._base import BaseBigQueryVectorStore

from langchain_google_community.feature_store.utils import (
    cast_proto_type,
    doc_match_filter,
)
from typing import Any, Dict, List, Optional, Type, Union
from google.cloud.aiplatform import base
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from google.cloud.aiplatform_v1beta1 import (
    FeatureOnlineStoreAdminServiceClient,
    FeatureOnlineStoreServiceClient,
)
from google.api_core.exceptions import (
    MethodNotImplemented,
    ServiceUnavailable,
)
from google.cloud.aiplatform_v1beta1.types import (
    NearestNeighborQuery,
    feature_online_store_service,
)
from google.cloud.aiplatform_v1beta1.types import (
    feature_online_store as feature_online_store_pb2,
)

logger = base.Logger(__name__)
# Constants for index creation
MIN_INDEX_ROWS = 5
INDEX_CHECK_INTERVAL = timedelta(seconds=60)
USER_AGENT_PREFIX = "FeatureStore"


class VertexFSVectorStore(BaseBigQueryVectorStore):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    online_store_name: Union[str, None] = None
    online_store_location: Union[str, None] = None
    online_store_type: Literal["bigtable", "optimized"] = "optimized"
    view_name: Union[str, None] = None
    cron_schedule: Union[str, None] = None
    min_node_count: int = 1
    max_node_count: int = 3
    cpu_utilization_target: int = 50
    algorithm_config: AlgorithmConfig = utils.TreeAhConfig()
    filter_columns: Optional[List[str]] = None
    crowding_column: Optional[str] = None
    distance_measure_type: Optional[DistanceMeasureType] = (
        utils.DistanceMeasureType.DOT_PRODUCT_DISTANCE
    )
    _user_agent: str = ""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        _, self._user_agent = get_user_agent(
            f"{USER_AGENT_PREFIX}-{type(self).__name__}"
        )
        self.online_store_name = self.online_store_name or self.dataset_name
        self.view_name = self.view_name or self.table_name
        self.location = self.location or self.location
        vertexai.init(project=self.project_id, location=self.location)

        api_endpoint = f"{self.location}-aiplatform.googleapis.com"
        self._admin_client = FeatureOnlineStoreAdminServiceClient(
            client_options={"api_endpoint": api_endpoint},
            client_info=get_client_info(module=self._user_agent),
        )
        self._online_store = self._create_online_store()
        self._search_client = self._get_search_client()
        self._feature_view = self._get_feature_view()
        logger.info(
            "VertexFSVectorStore initialized with Feature Store "
            f"{self.online_store_type} Vector Search. \n"
            "Optional batch serving available via .get_big_query_vector_store() method."
        )

    def _validate_bq_existing_source(
        self,
        project_id_param,
        dataset_param,
        table_param,
    ):
        bq_uri_split = self._feature_view.gca_resource.big_query_source.uri.split(".")  # type: ignore[union-attr]
        project_id = bq_uri_split[0].replace("bq://", "")
        dataset = bq_uri_split[1]
        table = bq_uri_split[2]
        try:
            assert project_id == project_id_param
            assert dataset == dataset_param
            assert table == table_param
        except AssertionError:
            error_message = (
                "The BQ table passed in input is"
                f"bq://{project_id_param}.{dataset_param}.{table_param} "
                f"while the BQ table linked to the feature view is "
                "{self._feature_view.gca_resource.big_query_source.uri}."
                "Make sure you are using the same table for the feature "
                "view."
            )
            raise AssertionError(error_message)

    def _get_search_client(self) -> FeatureOnlineStoreServiceClient:
        gca_resource = self._online_store.gca_resource
        endpoint = gca_resource.dedicated_serving_endpoint.public_endpoint_domain_name
        return FeatureOnlineStoreServiceClient(
            client_options={"api_endpoint": endpoint}
        )

    def _wait_until_dummy_query_success(self, timeout_seconds: int = 6000):
        """
        Waits until a dummy query succeeds, indicating the system is ready.
        """
        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutExpired(
                    "Timeout of {} seconds exceeded".format(timeout_seconds),
                    timeout=timeout_seconds,
                )
            try:
                return self._search_embedding(
                    embedding=[1] * self.embedding_dimension,  # type: ignore[operator]
                    k=1,
                )
            except ServiceUnavailable:
                logger.info(
                    "DNS certificates are being propagated,"
                    " waiting for 10 seconds.  "
                )
                time.sleep(10)
            except MethodNotImplemented as e:
                if e.args and "Received http2 header with status" in e.args[0]:
                    logger.info(
                        "DNS certificates are being propagated,"
                        " waiting for 10 seconds.  "
                    )
                    time.sleep(10)
                else:
                    raise

    def sync(self):
        """Sync the data from the Big Query source into the Executor source"""
        self._feature_view = self._create_feature_view()
        self._validate_bq_existing_source(
            project_id_param=self.project_id,
            dataset_param=self.dataset_name,
            table_param=self.table_name,
        )
        sync_response = self._admin_client.sync_feature_view(
            feature_view=(
                f"projects/{self.project_id}/"
                f"locations/{self.location}"
                f"/featureOnlineStores/{self.online_store_name}"
                f"/featureViews/{self.view_name}"
            )
        )
        while True:
            feature_view_sync = self._admin_client.get_feature_view_sync(
                name=sync_response.feature_view_sync
            )
            if feature_view_sync.run_time.end_time.seconds > 0:
                status = (
                    "Succeed" if feature_view_sync.final_status.code == 0 else "Failed"
                )
                logger.info(f"Sync {status} for {feature_view_sync.name}.")
                break
            else:
                logger.info("Sync ongoing, waiting for 30 seconds.")
            time.sleep(30)

        self._wait_until_dummy_query_success()

    def _similarity_search_by_vectors_with_scores_and_embeddings(
        self,
        embeddings: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        batch_size: Union[int, None] = None,
        **kwargs,
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
        output = []
        for query_embedding in embeddings:
            documents = []
            results = self._search_embedding(embedding=query_embedding, k=k, **kwargs)

            for result in results:
                metadata, embedding = {}, None

                for feature in result.entity_key_values.key_values.features:
                    if feature.name not in [
                        self.text_embedding_field,
                        self.content_field,
                    ]:
                        dict_values = proto.Message.to_dict(feature.value)
                        col_type, value = next(iter(dict_values.items()))
                        value = cast_proto_type(column=col_type, value=value)
                        metadata[feature.name] = value
                    if feature.name == self.text_embedding_field:
                        embedding = feature.value.double_array_value.values
                    if feature.name == self.content_field:
                        dict_values = proto.Message.to_dict(feature.value)
                        content = list(dict_values.values())[0]
                if filter is not None and not doc_match_filter(
                    document=metadata, filter=filter
                ):
                    continue
                documents.append(
                    [
                        Document(
                            page_content=content,
                            metadata=metadata,
                        ),
                        result.distance,
                        embedding,
                    ]
                )
            output.append(documents)
        return output

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
        output = []
        if ids is None:
            raise ValueError(
                "Feature Store executor doesn't support search by filter " "only"
            )
        for id in ids:
            with telemetry.tool_context_manager(self._user_agent):
                result = self._feature_view.read(key=[id])  # type: ignore[union-attr]
                metadata, content = {}, None
                for feature in result.to_dict()["features"]:
                    if feature["name"] not in [
                        self.text_embedding_field,
                        self.content_field,
                    ]:
                        metadata[feature["name"]] = list(feature["value"].values())[0]
                    if feature["name"] == self.content_field:
                        content = list(feature["value"].values())[0]
                if filter is not None and not doc_match_filter(
                    document=metadata, filter=filter
                ):
                    continue
                output.append(
                    Document(
                        page_content=str(content),
                        metadata=metadata,
                    )
                )
        return output

    def search_neighbors_by_ids(
        self, ids: List[str], filter: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[Document]:
        """Searches for neighboring entities in a Vertex Feature Store based on
            their IDs and optional filter on metadata

        Args:
            ids: A list of string identifiers representing the entities to search for.
            filter: (Optional) A dictionary specifying filter criteria for document
                    on metadata properties, e.g.
                                {
                                    "str_property": "foo",
                                    "int_property": 123
                                }
        """
        output = []
        if ids is None:
            raise ValueError(
                "Feature Store executor doesn't support search by filter " "only"
            )
        for entity_id in ids:
            results = self._search_embedding(entity_id=entity_id, **kwargs)
            for result in results:
                metadata, embedding = {}, None
                for feature in result.entity_key_values.key_values.features:
                    if feature.name not in [
                        self.text_embedding_field,
                        self.content_field,
                    ]:
                        dict_values = proto.Message.to_dict(feature.value)
                        metadata[feature.name] = list(dict_values.values())[0]
                    if feature.name == self.text_embedding_field:
                        embedding = feature.value.double_array_value
                    if feature.name == self.content_field:
                        dict_values = proto.Message.to_dict(feature.value)
                        content = list(dict_values.values())[0]
                if filter is not None and not doc_match_filter(
                    document=metadata, filter=filter
                ):
                    continue
                output.append(
                    [
                        Document(
                            page_content=content,
                            metadata=metadata,
                        ),
                        result.distance,
                        embedding,
                    ]
                )

        return output  # type: ignore[return-value]

    def _search_embedding(
        self,
        embedding: Optional[Any] = None,
        entity_id: Optional[str] = None,
        k: int = 5,
        string_filters: Optional[List[NearestNeighborQuery.StringFilter]] = None,
        per_crowding_attribute_neighbor_count: Optional[int] = None,
        approximate_neighbor_candidates: Optional[int] = None,
        leaf_nodes_search_fraction: Optional[float] = None,
    ) -> MutableSequence[Any]:
        if embedding:
            embedding = NearestNeighborQuery.Embedding(value=embedding)
        query = NearestNeighborQuery(
            entity_id=entity_id,
            embedding=embedding,
            neighbor_count=k,
            string_filters=string_filters,
            per_crowding_attribute_neighbor_count=per_crowding_attribute_neighbor_count,
            parameters={
                "approximate_neighbor_candidates": approximate_neighbor_candidates,
                "leaf_nodes_search_fraction": leaf_nodes_search_fraction,
            },
        )
        with telemetry.tool_context_manager(self._user_agent):
            result = self._search_client.search_nearest_entities(
                request=feature_online_store_service.SearchNearestEntitiesRequest(
                    feature_view=self._feature_view.gca_resource.name,  # type: ignore[union-attr]
                    query=query,
                    return_full_entity=True,  # returning entities with metadata
                )
            )
        return result.nearest_neighbors.neighbors

    def _create_online_store(self) -> FeatureOnlineStore:
        # Search for existing Online store
        stores_list = FeatureOnlineStore.list(
            project=self.project_id, location=self.location
        )
        for store in stores_list:
            if store.name == self.online_store_name:
                return store

        logger.info("Creating feature store online store")
        # Create it otherwise
        if self.online_store_type == "bigtable":
            online_store_config = feature_online_store_pb2.FeatureOnlineStore(
                bigtable=feature_online_store_pb2.FeatureOnlineStore.Bigtable(
                    auto_scaling=feature_online_store_pb2.FeatureOnlineStore.Bigtable.AutoScaling(
                        min_node_count=self.min_node_count,
                        max_node_count=self.max_node_count,
                        cpu_utilization_target=self.cpu_utilization_target,
                    )
                ),
                embedding_management=feature_online_store_pb2.FeatureOnlineStore.EmbeddingManagement(
                    enabled=True
                ),
            )
            create_store_lro = self._admin_client.create_feature_online_store(
                parent=f"projects/{self.project_id}/locations/{self.location}",
                feature_online_store_id=self.online_store_name,
                feature_online_store=online_store_config,
            )
            logger.info(create_store_lro.result())
        elif self.online_store_type == "optimized":
            online_store_config = feature_online_store_pb2.FeatureOnlineStore(
                optimized=feature_online_store_pb2.FeatureOnlineStore.Optimized()
            )
            create_store_lro = self._admin_client.create_feature_online_store(
                parent=f"projects/{self.project_id}/locations/{self.location}",
                feature_online_store_id=self.online_store_name,
                feature_online_store=online_store_config,
            )
            logger.info(create_store_lro.result())
            logger.info(create_store_lro.result())

        else:
            raise ValueError(
                f"{self.online_store_type} not allowed. "
                f"Accepted values are 'bigtable' or 'optimized'."
            )
        stores_list = FeatureOnlineStore.list(
            project=self.project_id, location=self.location
        )
        for store in stores_list:
            if store.name == self.online_store_name:
                return store

    def _create_feature_view(self) -> FeatureView:
        fv = self._get_feature_view()
        if fv:
            return fv
        else:
            big_query_source = FeatureViewBigQuerySource(
                uri=f"bq://{self._full_table_id}",
                entity_id_columns=[self.doc_id_field],
            )
            index_config = utils.IndexConfig(
                embedding_column=self.text_embedding_field,
                crowding_column=self.crowding_column,
                filter_columns=self.filter_columns,
                dimensions=self.embedding_dimension,
                distance_measure_type=self.distance_measure_type,
                algorithm_config=self.algorithm_config,
            )
            return self._online_store.create_feature_view(
                name=self.view_name,
                source=big_query_source,
                sync_config=self.cron_schedule,
                index_config=index_config,
                project=self.project_id,
                location=self.location,
            )

    def _get_feature_view(self) -> FeatureView | None:
        # Search for existing Feature view
        fv_list = FeatureView.list(
            feature_online_store_id=self._online_store.gca_resource.name
        )
        for fv in fv_list:
            if fv.name == self.view_name:
                return fv
        return None
    @classmethod
    def from_texts(
        cls: Type["VertexFSVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "VertexFSVectorStore":
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
        vs_obj = VertexFSVectorStore(embedding=embedding, **kwargs)
        vs_obj.add_texts(texts, metadatas)
        return vs_obj

    def get_big_query_vector_store(
            self,
            **kwargs
    ) -> BaseBigQueryVectorStore:
        from langchain_google_community.feature_store.bigquery import \
            BigQueryVectorStore

        base_params = self.dict(include=BaseBigQueryVectorStore.__fields__.keys())
        all_params = {**base_params, **kwargs}
        bq_obj = BigQueryVectorStore(**all_params)
        return bq_obj

