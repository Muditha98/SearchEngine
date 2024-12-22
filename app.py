# document_analysis/app.py

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from wordcloud import WordCloud

from core import Config, TextAnalyzer, ConfigurationError
from search import EnhancedSearchClient, DocumentStorage, DocumentProcessor
from graph import GraphBuilder, GraphAnalyzer, GraphVisualizer
from database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UIComponents:
    """UI Component class for reusable Streamlit components"""
    
    @staticmethod
    def setup_page_config():
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Document Analysis System",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Add custom CSS
        st.markdown("""
            <style>
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .stAlert {
                padding: 1rem;
                margin: 1rem 0;
            }
            .search-result {
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin: 1rem 0;
                border: 1px solid #e0e0e0;
                background: white;
            }
            .metric-card {
                background: white;
                padding: 1rem;
                border-radius: 0.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_sidebar(config: Config) -> Tuple[Dict[str, Any], bool]:
        """Render sidebar configuration and return settings"""
        with st.sidebar:
            st.header("🛠️ Configuration")

            # Service status
            st.subheader("Service Status")
            services = {
                "Azure Search": bool(config.SEARCH_ENDPOINT and config.SEARCH_API_KEY),
                "Cognitive Services": bool(config.COGNITIVE_SERVICE_ENDPOINT),
                "Document Storage": bool(config.BLOB_CONNECTION_STRING),
                "Graph Database": bool(config.GREMLIN_ENDPOINT)
            }

            for service, status in services.items():
                if status:
                    st.success(f"✅ {service}")
                else:
                    st.error(f"❌ {service}")

            # Search settings
            st.subheader("Search Settings")
            settings = {
                "search_type": st.selectbox(
                    "Search Type",
                    ["Semantic Search", "Full-Text Search"],
                    help="Semantic search understands context better"
                ),
                "result_count": st.slider(
                    "Number of Results",
                    min_value=5,
                    max_value=50,
                    value=10
                ),
                "time_filter": st.selectbox(
                    "Time Filter",
                    ["All Time", "Past Day", "Past Week", "Past Month", "Past Year"]
                )
            }

            # Advanced options
            with st.expander("Advanced Options"):
                settings.update({
                    "search_fields": st.multiselect(
                        "Search Fields",
                        ["content", "organizations", "locations", "keyphrases"],
                        default=["content"]
                    ),
                    "enable_cache": st.checkbox(
                        "Enable Caching",
                        value=True,
                        help="Cache results for better performance"
                    )
                })

            return settings, all(services.values())

    @staticmethod
    def render_document_view(doc: Dict[str, Any], analysis: Dict[str, Any]):
        """Render document view with analysis"""
        st.markdown(f"### 📄 {doc.get('metadata_storage_name', 'Untitled')}")

        tabs = st.tabs(["📝 Summary", "🎯 Analysis", "📊 Metrics", "📑 Content"])

        with tabs[0]:  # Summary tab
            st.markdown("#### Document Summary")
            st.markdown(analysis['analysis']['summary'])
            
            with st.expander("Document Metadata"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Created:**", UIComponents.format_date(
                        doc.get('metadata_creation_date', 'N/A')
                    ))
                    st.write("**Type:**", doc.get('metadata_storage_content_type', 'N/A'))
                with col2:
                    st.write("**Modified:**", UIComponents.format_date(
                        doc.get('metadata_storage_last_modified', 'N/A')
                    ))
                    st.write("**Language:**", analysis['analysis']['language'])

        with tabs[1]:  # Analysis tab
            UIComponents.render_analysis_tab(analysis)

        with tabs[2]:  # Metrics tab
            UIComponents.render_metrics_tab(analysis)

        with tabs[3]:  # Content tab
            content = doc.get("merged_content") or doc.get("content", "")
            if content:
                with st.expander("View Full Content"):
                    st.markdown(content)
            else:
                st.info("No content available")

    @staticmethod
    def render_analysis_tab(analysis: Dict[str, Any]):
        """Render analysis tab content"""
        st.markdown("#### Sentiment Analysis")
        sentiment = analysis['analysis']['sentiment']
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment['scores']['positive'] * 100,
            title={'text': f"Sentiment: {sentiment['overall'].title()}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ]
            }
        ))
        st.plotly_chart(fig)

        # Key entities and phrases
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Key Phrases")
            for phrase in analysis['analysis']['key_phrases']:
                st.markdown(f"- {phrase}")

        with col2:
            st.markdown("#### Named Entities")
            for entity in analysis['analysis']['entities']:
                confidence = entity['confidence_score'] * 100
                st.markdown(
                    f"- **{entity['text']}** ({entity['category']}) - "
                    f"{confidence:.1f}% confidence"
                )

    @staticmethod
    def render_metrics_tab(analysis: Dict[str, Any]):
        """Render metrics tab content"""
        st.markdown("#### Document Metrics")
        
        metrics = analysis['metrics']
        cols = st.columns(4)
        
        with cols[0]:
            st.metric("Words", metrics['word_count'])
        with cols[1]:
            st.metric("Organizations", metrics['entity_count']['organizations'])
        with cols[2]:
            st.metric("Locations", metrics['entity_count']['locations'])
        with cols[3]:
            st.metric("Key Phrases", metrics['entity_count']['keyphrases'])

    @staticmethod
    def render_graph_view(
        graph_builder: GraphBuilder,
        graph_analyzer: GraphAnalyzer,
        graph_visualizer: GraphVisualizer
    ):
        """Render graph visualization view"""
        st.subheader("🕸️ Document Relationship Graph")

        col1, col2 = st.columns([2, 1])

        with col1:
            metrics = graph_analyzer.calculate_metrics()
            net = graph_visualizer.create_network_visualization(
                list(graph_builder.nodes.values()),
                graph_builder.edges,
                metrics
            )

            # Save and display graph
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=800)

        with col2:
            st.markdown("### 📊 Graph Analytics")

            # Display key metrics
            st.metric("Graph Density", f"{metrics['density']:.2f}")
            st.metric("Average Clustering", f"{metrics['avg_clustering']:.2f}")

            # Show communities
            communities = graph_analyzer.find_communities()
            st.markdown(f"**Communities Detected:** {len(communities)}")

            # Central nodes
            central_nodes = graph_analyzer.find_central_nodes()
            if central_nodes:
                st.markdown("**Most Central Nodes:**")
                for node, centrality in central_nodes:
                    st.markdown(f"- {node}: {centrality:.2f}")

    @staticmethod
    def render_analytics_dashboard(results: List[Dict[str, Any]]):
        """Render analytics dashboard"""
        st.subheader("📈 Analytics Dashboard")

        # Process data
        df = pd.DataFrame(results)
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            sentiment_fig = px.pie(
                df['sentiment'].value_counts(),
                values='sentiment',
                names=df.index,
                title="Sentiment Distribution"
            )
            st.plotly_chart(sentiment_fig)

        with col2:
            # Entity distribution
            entities_df = pd.DataFrame([
                entity for doc in results
                for entity in doc['entities']
            ])
            if not entities_df.empty:
                entity_fig = px.bar(
                    entities_df['category'].value_counts(),
                    title="Entity Type Distribution"
                )
                st.plotly_chart(entity_fig)

        # Timeline analysis
        dates = [
            datetime.strptime(doc['created_date'], "%Y-%m-%dT%H:%M:%SZ")
            for doc in results if 'created_date' in doc
        ]
        if dates:
            timeline_df = pd.DataFrame({'date': dates})
            timeline_df['month'] = timeline_df['date'].dt.to_period('M')
            timeline_fig = px.line(
                timeline_df['month'].value_counts().sort_index(),
                title="Document Timeline"
            )
            st.plotly_chart(timeline_fig)

    @staticmethod
    def format_date(date_str: str) -> str:
        """Format date string"""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
            return date_obj.strftime("%B %d, %Y")
        except:
            return date_str

async def initialize_services(config: Config):
    """Initialize all required services"""
    try:
        text_analyzer = TextAnalyzer(config)
        search_client = EnhancedSearchClient(config)
        doc_storage = DocumentStorage(config)
        doc_processor = DocumentProcessor(text_analyzer, doc_storage)
        db_manager = DatabaseManager(config)
        
        return text_analyzer, search_client, doc_storage, doc_processor, db_manager
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

async def main():
    """Main application entry point"""
    try:
        # Setup UI
        UIComponents.setup_page_config()

        # Initialize configuration
        try:
            config = Config.from_env()
        except ConfigurationError as e:
            st.error(f"Configuration error: {str(e)}")
            st.info("Please check your .env file and Azure settings.")
            return

        # Initialize sidebar
        settings, services_ok = UIComponents.render_sidebar(config)

        if not services_ok:
            st.error("Some required services are not configured. Please check the sidebar.")
            return

        # Initialize services
        try:
            services = await initialize_services(config)
            text_analyzer, search_client, doc_storage, doc_processor, db_manager = services
        except Exception as e:
            st.error(f"Failed to initialize services: {str(e)}")
            return

        # Main search interface
        st.title("🔍 Document Search & Analysis")

        search_query = st.text_input(
            "Enter your search query",
            placeholder="Type your search query here..."
        )

        view_type = st.radio(
            "Select View",
            ["Document View", "Graph View", "Analytics Dashboard"],
            horizontal=True
        )

        if search_query:
            async with db_manager:
                try:
                    # Perform search
                    with st.spinner("🔍 Searching documents..."):
                        results = await search_client.search_documents(
                            query=search_query,
                            search_type=settings['search_type'].lower(),
                            top=settings['result_count'],
                            fields=settings['search_fields']
                        )

                    if results:
                        st.success(f"Found {len(results)} results")

                        if view_type == "Document View":
                            for doc in results:
                                with st.container():
                                    st.markdown("---")
                                    analysis = await doc_processor.process_document(doc)
                                    UIComponents.render_document_view(doc, analysis)

                        elif view_type == "Graph View":
                            graph_builder = GraphBuilder()
                            graph_analyzer = GraphAnalyzer()
                            graph_visualizer = GraphVisualizer()

                            for doc in results:
                                analysis = await doc_processor.process_document(doc)
                                graph_builder.add_document_node(
                                    doc['metadata_storage_name'],
                                    doc.get('metadata', {})
                                )
                                
                            UIComponents.render_graph_view(
                                graph_builder,
                                graph_analyzer,
                                graph_visualizer
                            )

                        else:  # Analytics Dashboard
                            analyses = [
                                await doc_processor.process_document(doc)
                                for doc in results
                            ]
                            UIComponents.render_analytics_dashboard(analyses)

                    else:
                        st.warning("No results found. Try modifying your search query.")

                except Exception as e:
                    logger.error(f"Application error: {str(e)}")
                    st.error("An error occurred while processing your request.")
                    
                    with st.expander("Error Details"):
                        st.code(str(e))

        else:
            # Welcome screen
            st.info("👋 Enter a search query to begin exploring documents.")
            
            with st.expander("📚 Quick Start Guide"):
                st.markdown("""
                ### Getting Started
                
                1. **Enter a search query** in the search box above
                2. **Choose a view type**:
                   - Document View: Detailed document analysis
                   - Graph View: Visualize document relationships
                   - Analytics Dashboard: Overall insights
                3. **Use the sidebar** to configure search settings
                
                ### Sample Queries
                - "financial reports from last quarter"
                - "technical documentation for project X"
                - "policy documents related to compliance"
                - "meeting minutes from 2024"
                - "project proposals with budget analysis"
                
                ### Features Available
                - 🔍 Semantic search capabilities
                - 📊 Document analytics and insights
                - 🎯 Entity recognition and extraction
                - 💭 Sentiment analysis
                - 🕸️ Document relationship visualization
                - 📈 Trend analysis and dashboards
                """)
            
            with st.expander("💡 Pro Tips"):
                st.markdown("""
                ### Search Tips
                
                1. **Use Semantic Search** for:
                   - Natural language queries
                   - Context-aware results
                   - Better relevance ranking
                
                2. **Use Full-Text Search** for:
                   - Exact phrase matching
                   - Simple keyword searches
                   - Faster results
                
                3. **Optimize Your Results**:
                   - Use specific keywords
                   - Apply time filters for recent documents
                   - Select relevant search fields
                   - Adjust the number of results
                """)

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("""
        An unexpected error occurred. Please try:
        1. Refreshing the page
        2. Checking your connection
        3. Verifying service configurations
        """)
        
        if st.checkbox("Show Error Details"):
            st.code(str(e))
    
    finally:
        # Cleanup resources if needed
        if 'db_manager' in locals():
            await db_manager.close()

if __name__ == "__main__":
    asyncio.run(main())