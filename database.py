# Part 4: Database Integration and Persistence Layer
# document_analysis/database.py

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging
import asyncio
import json
from datetime import datetime

from gremlin_python.driver import client, serializer, resultset
from gremlin_python.driver.protocol import GremlinServerError
from gremlin_python.process.graph_traversal import __, GraphTraversalSource
from gremlin_python.process.traversal import T, Direction, Cardinality
from azure.cosmos.aio import CosmosClient
from azure.core.exceptions import AzureError

from core import Config  # Import from Part 1
from graph import Node, Edge  # Import from Part 3

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class GremlinQuery:
    """Data class for Gremlin queries"""
    query: str
    bindings: Dict[str, Any]

class GremlinDatabaseClient:
    """Enhanced Gremlin database client with async support"""
    
    def __init__(self, config: Config):
        """Initialize the Gremlin client with configuration"""
        self.config = config
        self.client = None
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Gremlin client with error handling"""
        try:
            self.client = client.Client(
                url=self.config.GREMLIN_ENDPOINT,
                traversal_source='g',
                username=f"/dbs/{self.config.DATABASE_NAME}/colls/{self.config.CONTAINER_NAME}",
                password=self.config.GREMLIN_KEY,
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gremlin client: {str(e)}")
            raise

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the Gremlin client connection"""
        if self.client:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.close
            )

    async def execute_query(
        self, 
        query: Union[str, GremlinQuery],
        retries: int = None
    ) -> List[Dict[str, Any]]:
        """Execute Gremlin query with retry logic"""
        if isinstance(query, GremlinQuery):
            gremlin_query = query.query
            bindings = query.bindings
        else:
            gremlin_query = query
            bindings = {}

        retries = retries if retries is not None else self.max_retries

        for attempt in range(retries):
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.submit(gremlin_query, bindings).all().result()
                )
                return result

            except GremlinServerError as e:
                logger.error(f"Gremlin query error (attempt {attempt + 1}): {str(e)}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))

class GraphPersistence:
    """Handle graph data persistence and retrieval"""
    
    def __init__(self, gremlin_client: GremlinDatabaseClient):
        """Initialize with Gremlin client"""
        self.gremlin_client = gremlin_client
        self.batch_size = 100

    async def store_graph(
        self, 
        nodes: List[Node], 
        edges: List[Edge]
    ) -> bool:
        """Store graph data in Gremlin database"""
        try:
            # Clear existing data
            await self.clear_graph()

            # Store nodes in batches
            for i in range(0, len(nodes), self.batch_size):
                batch = nodes[i:i + self.batch_size]
                await self._store_node_batch(batch)

            # Store edges in batches
            for i in range(0, len(edges), self.batch_size):
                batch = edges[i:i + self.batch_size]
                await self._store_edge_batch(batch)

            return True

        except Exception as e:
            logger.error(f"Error storing graph: {str(e)}")
            return False

    async def clear_graph(self) -> None:
        """Clear all graph data"""
        query = GremlinQuery(
            query="g.V().drop()",
            bindings={}
        )
        await self.gremlin_client.execute_query(query)

    async def _store_node_batch(self, nodes: List[Node]) -> None:
        """Store a batch of nodes"""
        for node in nodes:
            query = self._build_node_query(node)
            await self.gremlin_client.execute_query(query)

    async def _store_edge_batch(self, edges: List[Edge]) -> None:
        """Store a batch of edges"""
        for edge in edges:
            query = self._build_edge_query(edge)
            await self.gremlin_client.execute_query(query)

    def _build_node_query(self, node: Node) -> GremlinQuery:
        """Build Gremlin query for node insertion"""
        properties = {
            k: v for k, v in node.properties.items()
            if v is not None
        }
        
        query = """
        g.addV(label)
            .property('id', id)
            .property('type', type)
            .property('group', group)
        """
        
        bindings = {
            "label": node.label,
            "id": node.id,
            "type": node.type,
            "group": node.group
        }

        # Add property assignments
        for key, value in properties.items():
            query += f".property('{key}', {key})"
            bindings[key] = value

        return GremlinQuery(query=query, bindings=bindings)

    def _build_edge_query(self, edge: Edge) -> GremlinQuery:
        """Build Gremlin query for edge insertion"""
        properties = {
            k: v for k, v in edge.properties.items()
            if v is not None
        }
        
        query = """
        g.V().hasId(source)
            .addE(label)
            .to(g.V().hasId(target))
            .property('type', type)
        """
        
        bindings = {
            "source": edge.source,
            "target": edge.target,
            "label": edge.label,
            "type": edge.type
        }

        # Add property assignments
        for key, value in properties.items():
            query += f".property('{key}', {key})"
            bindings[key] = value

        return GremlinQuery(query=query, bindings=bindings)

class GraphQuerying:
    """Handle complex graph queries and traversals"""
    
    def __init__(self, gremlin_client: GremlinDatabaseClient):
        """Initialize with Gremlin client"""
        self.gremlin_client = gremlin_client

    async def find_connected_documents(
        self, 
        entity_id: str, 
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Find documents connected to a given entity"""
        query = GremlinQuery(
            query="""
            g.V(entityId)
                .repeat(both().simplePath())
                .times(maxDepth)
                .path()
                .by(valueMap().with(WithOptions.tokens))
            """,
            bindings={
                "entityId": entity_id,
                "maxDepth": max_depth
            }
        )
        
        try:
            return await self.gremlin_client.execute_query(query)
        except Exception as e:
            logger.error(f"Error finding connected documents: {str(e)}")
            return []

    async def find_similar_documents(
        self, 
        doc_id: str, 
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Find documents sharing similar entities"""
        query = GremlinQuery(
            query="""
            g.V(docId).as('doc')
                .out().aggregate('entities')
                .in_().where(neq('doc'))
                .project('id', 'similarity', 'properties')
                    .by(id)
                    .by(
                        out().where(within('entities')).count()
                        .multiply(1.0)
                        .divide(out().count())
                    )
                    .by(valueMap())
                .where(__.select('similarity').is(gte(minSimilarity)))
                .order().by('similarity', decr)
            """,
            bindings={
                "docId": doc_id,
                "minSimilarity": min_similarity
            }
        )
        
        try:
            return await self.gremlin_client.execute_query(query)
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []

    async def get_entity_relationships(
        self, 
        entity_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get all relationships for a specific entity"""
        try:
            # Get incoming relationships
            in_query = GremlinQuery(
                query="""
                g.V(entityId)
                    .inE()
                    .project('edge', 'vertex')
                    .by(valueMap().with(WithOptions.tokens))
                    .by(outV().valueMap().with(WithOptions.tokens))
                """,
                bindings={"entityId": entity_id}
            )
            incoming = await self.gremlin_client.execute_query(in_query)

            # Get outgoing relationships
            out_query = GremlinQuery(
                query="""
                g.V(entityId)
                    .outE()
                    .project('edge', 'vertex')
                    .by(valueMap().with(WithOptions.tokens))
                    .by(inV().valueMap().with(WithOptions.tokens))
                """,
                bindings={"entityId": entity_id}
            )
            outgoing = await self.gremlin_client.execute_query(out_query)

            return {
                "incoming": incoming,
                "outgoing": outgoing
            }
            
        except Exception as e:
            logger.error(f"Error getting entity relationships: {str(e)}")
            return {"incoming": [], "outgoing": []}

class GraphCache:
    """Cache layer for graph queries"""
    
    def __init__(self, max_size: int = 1000):
        """Initialize cache with maximum size"""
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None

    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with LRU eviction"""
        if len(self.cache) >= self.max_size:
            # Evict least recently used item
            oldest_key = min(
                self.access_times.items(),
                key=lambda x: x[1]
            )[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = value
        self.access_times[key] = datetime.now()

    async def clear(self) -> None:
        """Clear the cache"""
        self.cache.clear()
        self.access_times.clear()

class DatabaseManager:
    """Manage database operations and connections"""
    
    def __init__(self, config: Config):
        """Initialize database manager"""
        self.gremlin_client = GremlinDatabaseClient(config)
        self.graph_persistence = GraphPersistence(self.gremlin_client)
        self.graph_querying = GraphQuerying(self.gremlin_client)
        self.cache = GraphCache()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close all database connections"""
        await self.gremlin_client.close()