# Part 3: Graph Analysis and Visualization Components
# document_analysis/graph.py

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import asyncio
from datetime import datetime

import networkx as nx
from gremlin_python.driver import client, serializer, resultset
from gremlin_python.driver.protocol import GremlinServerError
import community
from pyvis.network import Network
import plotly.graph_objects as go

from core import Config  # Import from Part 1

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Node:
    """Data class for graph nodes"""
    id: str
    label: str
    type: str
    properties: Dict[str, Any]
    group: str

@dataclass
class Edge:
    """Data class for graph edges"""
    source: str
    target: str
    label: str
    type: str
    properties: Dict[str, Any]

class GraphBuilder:
    """Build and manage graph structures"""
    
    def __init__(self):
        """Initialize the graph builder"""
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.graph = nx.Graph()

    def add_document_node(
        self, 
        doc_id: str, 
        metadata: Dict[str, Any]
    ) -> None:
        """Add a document node to the graph"""
        node = Node(
            id=doc_id,
            label=doc_id,
            type='document',
            properties={
                'created_date': metadata.get('creation_date'),
                'modified_date': metadata.get('last_modified'),
                'file_type': metadata.get('content_type'),
                'size': metadata.get('size')
            },
            group='documents'
        )
        self.nodes[doc_id] = node
        self._add_node_to_networkx(node)

    def add_entity_node(
        self, 
        entity_text: str, 
        entity_type: str, 
        confidence: float
    ) -> str:
        """Add an entity node to the graph"""
        entity_id = f"{entity_type}_{entity_text.replace(' ', '_')}"
        
        if entity_id not in self.nodes:
            node = Node(
                id=entity_id,
                label=entity_text,
                type=entity_type,
                properties={'confidence': confidence},
                group=f"{entity_type}s"
            )
            self.nodes[entity_id] = node
            self._add_node_to_networkx(node)
            
        return entity_id

    def add_edge(
        self, 
        source: str, 
        target: str, 
        label: str, 
        properties: Dict[str, Any] = None
    ) -> None:
        """Add an edge to the graph"""
        if source in self.nodes and target in self.nodes:
            edge = Edge(
                source=source,
                target=target,
                label=label,
                type=label.lower(),
                properties=properties or {}
            )
            self.edges.append(edge)
            self._add_edge_to_networkx(edge)

    def _add_node_to_networkx(self, node: Node) -> None:
        """Add a node to the NetworkX graph"""
        self.graph.add_node(
            node.id,
            label=node.label,
            type=node.type,
            group=node.group,
            **node.properties
        )

    def _add_edge_to_networkx(self, edge: Edge) -> None:
        """Add an edge to the NetworkX graph"""
        self.graph.add_edge(
            edge.source,
            edge.target,
            label=edge.label,
            type=edge.type,
            **edge.properties
        )

class GraphAnalyzer:
    """Advanced graph analysis capabilities"""
    
    def __init__(self, graph: nx.Graph):
        """Initialize with a NetworkX graph"""
        self.graph = graph
        self.cache = {}

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate various graph metrics"""
        try:
            metrics = {
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'avg_clustering': nx.average_clustering(self.graph),
            }

            # Calculate centrality measures
            metrics['degree_centrality'] = nx.degree_centrality(self.graph)
            metrics['betweenness_centrality'] = nx.betweenness_centrality(self.graph)
            
            # Calculate connected components
            connected_components = list(nx.connected_components(self.graph))
            metrics['connected_components'] = [
                list(component) for component in connected_components
            ]
            
            # Calculate additional metrics for connected graphs
            largest_component = max(connected_components, key=len)
            subgraph = self.graph.subgraph(largest_component)
            
            if nx.is_connected(subgraph):
                metrics['diameter'] = nx.diameter(subgraph)
                metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph)

            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating graph metrics: {str(e)}")
            return self._get_default_metrics()

    def find_communities(self) -> List[Set[str]]:
        """Detect communities using the Louvain method"""
        try:
            partition = community.best_partition(self.graph)
            communities: Dict[int, Set[str]] = {}
            
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = set()
                communities[community_id].add(node)
                
            return list(communities.values())
            
        except Exception as e:
            logger.error(f"Error detecting communities: {str(e)}")
            return []

    def find_central_nodes(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the most central nodes using various centrality measures"""
        try:
            # Calculate different centrality measures
            degree_cent = nx.degree_centrality(self.graph)
            between_cent = nx.betweenness_centrality(self.graph)
            eigen_cent = nx.eigenvector_centrality(self.graph)
            
            # Combine centrality measures with weights
            combined_centrality = {}
            for node in self.graph.nodes():
                combined_centrality[node] = (
                    0.4 * degree_cent[node] +
                    0.4 * between_cent[node] +
                    0.2 * eigen_cent[node]
                )
            
            # Sort and return top K nodes
            return sorted(
                combined_centrality.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding central nodes: {str(e)}")
            return []

    def find_similar_documents(
        self, 
        doc_id: str, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar documents based on shared entities"""
        try:
            if doc_id not in self.graph:
                return []

            similarities = {}
            doc_entities = set(
                n for n in self.graph.neighbors(doc_id)
                if self.graph.nodes[n]['type'] != 'document'
            )

            for node in self.graph.nodes():
                if (node != doc_id and 
                    self.graph.nodes[node]['type'] == 'document'):
                    
                    other_entities = set(
                        n for n in self.graph.neighbors(node)
                        if self.graph.nodes[n]['type'] != 'document'
                    )
                    
                    # Calculate Jaccard similarity
                    similarity = len(doc_entities & other_entities) / \
                               len(doc_entities | other_entities) if doc_entities else 0
                    
                    similarities[node] = similarity

            return sorted(
                similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []

    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default metrics for error cases"""
        return {
            'node_count': 0,
            'edge_count': 0,
            'density': 0.0,
            'avg_clustering': 0.0,
            'degree_centrality': {},
            'betweenness_centrality': {},
            'connected_components': []
        }

class GraphVisualizer:
    """Create interactive graph visualizations"""
    
    def __init__(self, height: str = "600px", width: str = "100%"):
        """Initialize visualizer with dimensions"""
        self.height = height
        self.width = width
        self.node_colors = {
            'document': '#4CAF50',
            'organization': '#2196F3',
            'location': '#FFC107',
            'person': '#9C27B0',
            'keyphrase': '#FF5722'
        }

    def create_network_visualization(
        self, 
        nodes: List[Node], 
        edges: List[Edge], 
        metrics: Dict[str, Any]
    ) -> Network:
        """Create an interactive network visualization"""
        net = Network(
            height=self.height,
            width=self.width,
            bgcolor="#ffffff",
            font_color="black"
        )

        # Configure physics
        net.force_atlas_2based(
            gravity=-50,
            central_gravity=0.01,
            spring_length=100,
            spring_strength=0.08,
            damping=0.4,
            overlap=0.75
        )

        # Add nodes
        degree_cent = metrics.get('degree_centrality', {})
        for node in nodes:
            size = 20 + (degree_cent.get(node.id, 0) * 50)
            color = self.node_colors.get(node.type, '#999999')
            
            net.add_node(
                node.id,
                label=node.label,
                title=self._create_node_tooltip(node),
                color=color,
                size=size,
                shape='dot' if node.type == 'document' else 'triangle'
            )

        # Add edges
        for edge in edges:
            net.add_edge(
                edge.source,
                edge.target,
                title=edge.label,
                color={'color': '#666666', 'opacity': 0.8},
                arrows='to',
                smooth={'type': 'curvedCW', 'roundness': 0.2}
            )

        return net

    def create_community_visualization(
        self, 
        communities: List[Set[str]], 
        graph: nx.Graph
    ) -> go.Figure:
        """Create a visualization of community structure"""
        node_colors = []
        node_text = []
        edge_x = []
        edge_y = []
        
        # Compute layout
        pos = nx.spring_layout(graph)
        
        # Process edges
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Process nodes
        node_x = []
        node_y = []
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Assign colors based on community
            for i, community in enumerate(communities):
                if node in community:
                    node_colors.append(f'rgb({50 + i*30}, {100 + i*20}, {150 + i*30})')
                    break
            
            # Create node tooltips
            node_text.append(f"Node: {node}<br>Type: {graph.nodes[node]['type']}")
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=10,
                color=node_colors,
                line_width=2
            )
        ))
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            plot_bgcolor='white'
        )
        
        return fig

    def _create_node_tooltip(self, node: Node) -> str:
        """Create detailed tooltip for node"""
        tooltip = f"Type: {node.type}<br>"
        tooltip += f"Label: {node.label}<br>"
        
        for key, value in node.properties.items():
            if value is not None:
                if isinstance(value, (datetime, float)):
                    tooltip += f"{key}: {value:.2f}<br>"
                else:
                    tooltip += f"{key}: {value}<br>"
                    
        return tooltip