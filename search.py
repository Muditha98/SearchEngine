# Part 2: Search and Document Processing Components
# document_analysis/search.py

from typing import Dict, List, Optional, Any, TypeVar
from datetime import datetime
import logging
from dataclasses import dataclass
import json
import asyncio

import aiohttp
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from azure.core.exceptions import AzureError

from core import Config, TextAnalyzer  # Import from Part 1

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Data class for search results"""
    document_id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    highlights: Dict[str, List[str]]

class EnhancedSearchClient:
    """Improved search client with better error handling and async support"""
    
    def __init__(self, config: Config):
        """Initialize the search client with configuration"""
        self.endpoint = config.SEARCH_ENDPOINT.rstrip('/')
        self.index_name = config.SEARCH_INDEX
        self.api_version = "2023-11-01"
        self.credential = AzureKeyCredential(config.SEARCH_API_KEY)
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        self.session = aiohttp.ClientSession()
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def search_documents(
        self,
        query: str,
        search_type: str = "semantic",
        top: int = 10,
        fields: Optional[List[str]] = None,
        filter_str: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform document search asynchronously with retries"""
        search_fields = fields or [
            "content", "merged_content", "text",
            "keyphrases", "organizations", "locations"
        ]

        body = {
            "search": query,
            "select": "*",
            "top": top,
            "searchFields": search_fields,
            "count": True,
            "queryType": "semantic" if search_type == "semantic" else "full",
            "highlight": search_fields,
            "highlightPreTag": "<mark>",
            "highlightPostTag": "</mark>"
        }

        if search_type == "semantic":
            body.update({
                "semanticConfiguration": "default",
                "queryLanguage": "en-us",
                "captions": "extractive",
                "answers": "extractive"
            })

        if filter_str:
            body["filter"] = filter_str

        # Implement retry logic
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{self.endpoint}/indexes/{self.index_name}/docs/search",
                    params={"api-version": self.api_version},
                    headers={
                        "Content-Type": "application/json",
                        "api-key": self.credential.key
                    },
                    json=body
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return [self._parse_search_result(doc) for doc in result.get("value", [])]
                    
            except aiohttp.ClientError as e:
                logger.error(f"Search request failed (attempt {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    def _parse_search_result(self, doc: Dict[str, Any]) -> SearchResult:
        """Parse raw search result into SearchResult object"""
        return SearchResult(
            document_id=doc.get("metadata_storage_name", ""),
            score=doc.get("@search.score", 0.0),
            content=doc.get("merged_content") or doc.get("content", ""),
            metadata={
                "creation_date": doc.get("metadata_creation_date"),
                "last_modified": doc.get("metadata_storage_last_modified"),
                "content_type": doc.get("metadata_storage_content_type"),
                "size": doc.get("metadata_storage_size"),
                "language": doc.get("metadata_language")
            },
            highlights=doc.get("@search.highlights", {})
        )

class DocumentStorage:
    """Enhanced document storage handling with async support"""
    
    def __init__(self, config: Config):
        """Initialize document storage with configuration"""
        self.blob_service_client = BlobServiceClient.from_connection_string(
            config.BLOB_CONNECTION_STRING
        )
        self.container_name = config.BLOB_CONTAINER_NAME
        self._container_client: Optional[ContainerClient] = None
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    @property
    def container_client(self) -> ContainerClient:
        """Lazy initialization of container client"""
        if self._container_client is None:
            self._container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
        return self._container_client

    async def get_document_content(self, blob_name: str) -> Optional[str]:
        """Retrieve document content from blob storage with retries"""
        blob_client = self.container_client.get_blob_client(blob_name)
        
        for attempt in range(self.max_retries):
            try:
                download_stream = await blob_client.download_blob()
                return await download_stream.content_as_text()
            
            except AzureError as e:
                logger.error(f"Error retrieving document (attempt {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    return None
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def get_document_metadata(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """Get document metadata from blob storage with retries"""
        blob_client = self.container_client.get_blob_client(blob_name)
        
        for attempt in range(self.max_retries):
            try:
                properties = await blob_client.get_blob_properties()
                return {
                    "content_type": properties.content_settings.content_type,
                    "size": properties.size,
                    "created_on": properties.creation_time,
                    "modified_on": properties.last_modified,
                    "metadata": properties.metadata,
                    "lease_state": properties.lease.state,
                    "archive_status": properties.archive_status
                }
            
            except AzureError as e:
                logger.error(f"Error retrieving metadata (attempt {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    return None
                await asyncio.sleep(self.retry_delay * (attempt + 1))

class DocumentProcessor:
    """Process and analyze documents"""
    
    def __init__(self, text_analyzer: TextAnalyzer, doc_storage: DocumentStorage):
        """Initialize with required components"""
        self.text_analyzer = text_analyzer
        self.doc_storage = doc_storage
        self.supported_formats = {
            'application/pdf', 'text/plain', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }

    async def process_document(self, search_result: SearchResult) -> Dict[str, Any]:
        """Process document and generate analysis"""
        try:
            # Get full content if not already present
            content = search_result.content
            if not content:
                content = await self.doc_storage.get_document_content(search_result.document_id)
                if not content:
                    raise ValueError(f"Could not retrieve content for document {search_result.document_id}")

            # Perform text analysis
            analysis = await self.text_analyzer.analyze_text(content)

            # Calculate additional metrics
            metrics = await self._calculate_metrics(search_result, content)

            return {
                "document_id": search_result.document_id,
                "analysis": analysis,
                "metrics": metrics,
                "metadata": search_result.metadata
            }

        except Exception as e:
            logger.error(f"Error processing document {search_result.document_id}: {str(e)}")
            return self._get_default_processing_result(search_result.document_id)

    async def _calculate_metrics(
        self, 
        search_result: SearchResult, 
        content: str
    ) -> Dict[str, Any]:
        """Calculate document metrics asynchronously"""
        word_count = len(content.split()) if content else 0
        
        # Get detailed metadata
        metadata = await self.doc_storage.get_document_metadata(search_result.document_id)
        
        return {
            "word_count": word_count,
            "character_count": len(content),
            "average_word_length": len(content) / word_count if word_count > 0 else 0,
            "entity_count": {
                "organizations": len(search_result.metadata.get("organizations", [])),
                "locations": len(search_result.metadata.get("locations", [])),
                "keyphrases": len(search_result.metadata.get("keyphrases", []))
            },
            "metadata": metadata or {}
        }

    def _get_default_processing_result(self, document_id: str) -> Dict[str, Any]:
        """Return default processing result for error cases"""
        return {
            "document_id": document_id,
            "analysis": self.text_analyzer._get_default_analysis_result(),
            "metrics": {
                "word_count": 0,
                "character_count": 0,
                "average_word_length": 0,
                "entity_count": {
                    "organizations": 0,
                    "locations": 0,
                    "keyphrases": 0
                },
                "metadata": {}
            },
            "metadata": {}
        }