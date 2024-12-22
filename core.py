# Part 1: Core Components - Configuration and Text Analysis
# document_analysis/core.py

import os
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any, TypeVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from azure.storage.blob import BlobServiceClient
import nltk
from nltk.tokenize import sent_tokenize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {str(e)}")

# Type aliases
T = TypeVar('T')
DocumentId = str
NodeId = str
EdgeId = str

class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass

@dataclass
class Config:
    """Configuration class with validation"""
    SEARCH_ENDPOINT: str
    SEARCH_INDEX: str
    SEARCH_API_KEY: str
    COGNITIVE_SERVICE_ENDPOINT: str
    COGNITIVE_SERVICE_KEY: str
    BLOB_CONNECTION_STRING: str
    BLOB_CONTAINER_NAME: str
    GREMLIN_ENDPOINT: str
    GREMLIN_KEY: str
    DATABASE_NAME: str
    CONTAINER_NAME: str

    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables with validation"""
        load_dotenv()
        
        required_vars = {
            "AZURE_SEARCH_ENDPOINT": "SEARCH_ENDPOINT",
            "AZURE_SEARCH_INDEX": "SEARCH_INDEX",
            "AZURE_SEARCH_API_KEY": "SEARCH_API_KEY",
            "AZURE_COGNITIVE_SERVICE_ENDPOINT": "COGNITIVE_SERVICE_ENDPOINT",
            "AZURE_COGNITIVE_SERVICE_KEY": "COGNITIVE_SERVICE_KEY",
            "AZURE_BLOB_CONNECTION_STRING": "BLOB_CONNECTION_STRING",
            "AZURE_BLOB_CONTAINER_NAME": "BLOB_CONTAINER_NAME",
            "COSMOS_GREMLIN_ENDPOINT": "GREMLIN_ENDPOINT",
            "COSMOS_GREMLIN_KEY": "GREMLIN_KEY",
            "COSMOS_DATABASE": "DATABASE_NAME",
            "COSMOS_CONTAINER": "CONTAINER_NAME"
        }

        config_dict = {}
        missing_vars = []

        for env_var, config_var in required_vars.items():
            value = os.getenv(env_var)
            if value is None:
                missing_vars.append(env_var)
            config_dict[config_var] = value or ""

        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        return cls(**config_dict)

    def validate(self) -> None:
        """Validate configuration values"""
        if not self.SEARCH_ENDPOINT.startswith(('http://', 'https://')):
            raise ConfigurationError("Invalid SEARCH_ENDPOINT URL format")
        
        if not self.COGNITIVE_SERVICE_ENDPOINT.startswith(('http://', 'https://')):
            raise ConfigurationError("Invalid COGNITIVE_SERVICE_ENDPOINT URL format")
        
        # Add more validation as needed
        
class TextAnalyzer:
    """Enhanced text analysis using Azure Cognitive Services"""
    
    def __init__(self, config: Config):
        """Initialize the TextAnalyzer with configuration"""
        self.client = TextAnalyticsClient(
            endpoint=config.COGNITIVE_SERVICE_ENDPOINT,
            credential=AzureKeyCredential(config.COGNITIVE_SERVICE_KEY)
        )
        self.max_chunk_size = 5000
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    async def analyze_text(
        self, 
        text: str, 
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze text asynchronously with better error handling"""
        try:
            if not text:
                raise ValueError("Empty text provided for analysis")

            # Detect language if not provided
            if not language:
                language_response = await self._detect_language(text)
                language = language_response.primary_language.iso6391_name

            # Process text in chunks if needed
            chunks = self._chunk_text(text)
            
            # Gather all analysis tasks
            analysis_tasks = [
                self._analyze_sentiment(chunks[0], language),
                self._extract_key_phrases(chunks[0], language),
                self._recognize_entities(chunks[0], language),
                self._generate_summary(chunks, language)
            ]

            # Execute all analysis tasks concurrently
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results and handle any exceptions
            sentiment, key_phrases, entities, summary = self._process_analysis_results(results)

            return {
                "language": language,
                "sentiment": sentiment,
                "key_phrases": key_phrases[:15],
                "entities": entities,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"Text analysis error: {str(e)}")
            return self._get_default_analysis_result()

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks"""
        if len(text) <= self.max_chunk_size:
            return [text]

        chunks = []
        sentences = sent_tokenize(text)
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunks.append('. '.join(current_chunk))

        return chunks

    async def _detect_language(self, text: str) -> Any:
        """Detect text language asynchronously with retries"""
        for attempt in range(self.max_retries):
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    lambda: next(self.client.detect_language([text]))
                )
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def _analyze_sentiment(
        self, 
        text: str, 
        language: str
    ) -> Dict[str, Any]:
        """Analyze sentiment asynchronously"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: next(self.client.analyze_sentiment(
                [text],
                language=language,
                show_opinion_mining=True
            ))
        )
        return {
            "overall": result.sentiment,
            "scores": {
                "positive": result.confidence_scores.positive,
                "neutral": result.confidence_scores.neutral,
                "negative": result.confidence_scores.negative
            }
        }

    async def _extract_key_phrases(
        self, 
        text: str, 
        language: str
    ) -> List[str]:
        """Extract key phrases asynchronously"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: next(self.client.extract_key_phrases(
                [text],
                language=language
            ))
        )
        return result.key_phrases

    async def _recognize_entities(
        self, 
        text: str, 
        language: str
    ) -> List[Dict[str, Any]]:
        """Recognize entities asynchronously"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: next(self.client.recognize_entities(
                [text],
                language=language
            ))
        )
        return [
            {
                "text": entity.text,
                "category": entity.category,
                "confidence_score": entity.confidence_score
            }
            for entity in result.entities
        ]

    async def _generate_summary(
        self, 
        chunks: List[str], 
        language: str
    ) -> str:
        """Generate text summary asynchronously"""
        summaries = []
        for chunk in chunks:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: next(self.client.extractive_summarization(
                    [chunk],
                    language=language,
                    max_sentence_count=3
                ))
            )
            summaries.extend([sent.text for sent in result.sentences])
        return " ".join(summaries[:5])

    def _process_analysis_results(
        self, 
        results: List[Any]
    ) -> Tuple[Dict[str, Any], List[str], List[Dict[str, Any]], str]:
        """Process analysis results and handle exceptions"""
        default_result = self._get_default_analysis_result()
        
        sentiment = results[0] if isinstance(results[0], dict) else default_result['sentiment']
        key_phrases = results[1] if isinstance(results[1], list) else default_result['key_phrases']
        entities = results[2] if isinstance(results[2], list) else default_result['entities']
        summary = results[3] if isinstance(results[3], str) else default_result['summary']
        
        return sentiment, key_phrases, entities, summary

    def _get_default_analysis_result(self) -> Dict[str, Any]:
        """Return default analysis result for error cases"""
        return {
            "language": "en",
            "sentiment": {
                "overall": "neutral",
                "scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
            },
            "key_phrases": [],
            "entities": [],
            "summary": "Error analyzing text"
        }