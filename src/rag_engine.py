"""
OKT-RAG Engine - Model-Agnostic, Observable RAG with Multi-Embedding Support.

Combines LightRAG's knowledge graph capabilities with:
- Multi-embedding storage for flexible retrieval
- Complete observability and logging
- Model-agnostic LLM layer
- Adaptive retrieval strategies
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from .connectors.base import Document
from .index_tracker import IndexTracker


class QueryMode(str, Enum):
    """LightRAG query modes + adaptive mode."""

    NAIVE = "naive"  # Simple vector similarity
    LOCAL = "local"  # Local context from knowledge graph
    GLOBAL = "global"  # Global patterns from knowledge graph
    HYBRID = "hybrid"  # Combines local and global
    ADAPTIVE = "adaptive"  # Auto-select based on query type


@dataclass
class QueryResult:
    """Result from a RAG query."""

    answer: str
    mode: QueryMode
    sources: list[dict]
    metadata: dict


@dataclass
class CitationVerification:
    """Verification result for a single citation."""

    citation_number: int
    statement: str
    source_file: str
    source_page: Optional[int]
    source_excerpt: str
    is_accurate: bool
    confidence: float  # 0.0 to 1.0
    explanation: str


@dataclass
class VerifiedQueryResult:
    """Result from a RAG query with automatic citation verification."""

    original_answer: str
    corrected_answer: str
    mode: QueryMode
    sources: list[dict]
    verification_log: list[dict]
    accuracy_rate: float
    removed_citations: list[int]
    metadata: dict


@dataclass
class VerificationResult:
    """Result from citation verification."""

    total_citations: int
    verified_count: int
    accurate_count: int
    inaccurate_count: int
    uncertain_count: int
    accuracy_rate: float
    verifications: list[CitationVerification]


class RAGEngine:
    """
    OKT-RAG Engine with multi-embedding and observability support.

    Combines LightRAG with:
    - MultiEmbeddingStore for multi-slot embeddings
    - RetrievalLogger for complete observability
    - LLMProvider for model-agnostic generation
    - AdaptiveRetriever for smart retrieval strategies
    """

    def __init__(
        self,
        working_dir: str | Path,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        enable_observability: bool = True,
        enable_multi_embedding: bool = True,
    ):
        """
        Initialize the OKT-RAG engine.

        Args:
            working_dir: Directory for LightRAG data storage.
            openai_api_key: OpenAI API key (uses env var if not provided).
            model_name: LLM model for generation.
            embedding_model: Model for text embeddings.
            enable_observability: Enable retrieval logging.
            enable_multi_embedding: Enable multi-embedding store.
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Set API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        self.model_name = model_name
        self.embedding_model = embedding_model
        self._rag = None
        self._initialized = False
        self._indexed_count = 0
        self._tracker = IndexTracker(self.working_dir)

        # OKT-RAG components
        self._enable_observability = enable_observability
        self._enable_multi_embedding = enable_multi_embedding

        # Initialize observability
        self._logger = None
        if enable_observability:
            self._init_observability()

        # Initialize multi-embedding store
        self._embedding_store = None
        self._embedding_providers = {}
        if enable_multi_embedding:
            self._init_multi_embedding()

        # Initialize LLM provider
        self._llm_provider = None
        self._init_llm_provider()

        # Initialize adaptive retriever
        self._adaptive_retriever = None
        if enable_multi_embedding:
            self._init_adaptive_retriever()

    def _init_observability(self) -> None:
        """Initialize observability components."""
        try:
            from .observability import RetrievalLogger

            log_path = self.working_dir / "logs" / "retrieval"
            self._logger = RetrievalLogger(
                storage_path=log_path,
                buffer_size=100,
                enabled=True,
            )
        except ImportError:
            self._logger = None

    def _init_multi_embedding(self) -> None:
        """Initialize multi-embedding store and providers."""
        try:
            from .embeddings import (
                MultiEmbeddingStore,
                EmbeddingSlot,
                OpenAIEmbeddingProvider,
            )

            # Default slots configuration
            slots = [
                EmbeddingSlot(
                    name="semantic",
                    provider_type="openai",
                    model="text-embedding-3-small",
                    dimension=1536,
                    weight=1.0,
                    description="Primary semantic embedding",
                ),
                EmbeddingSlot(
                    name="semantic_fast",
                    provider_type="openai",
                    model="text-embedding-3-small",
                    dimension=512,
                    weight=0.8,
                    description="Fast semantic (Matryoshka 512D)",
                ),
            ]

            self._embedding_store = MultiEmbeddingStore(
                working_dir=self.working_dir / "embeddings",
                slots=slots,
            )

            # Register providers for each slot
            for slot in slots:
                if slot.provider_type == "openai":
                    provider = OpenAIEmbeddingProvider(
                        model=slot.model,
                        dimension=slot.dimension,
                    )
                    self._embedding_store.register_provider(slot.name, provider)
                    self._embedding_providers[slot.name] = provider

        except ImportError:
            self._embedding_store = None

    def _init_llm_provider(self) -> None:
        """Initialize LLM provider."""
        try:
            from .llm import OpenAILLMProvider

            self._llm_provider = OpenAILLMProvider(model=self.model_name)
        except ImportError:
            self._llm_provider = None

    def _init_adaptive_retriever(self) -> None:
        """Initialize adaptive retriever."""
        try:
            from .retrieval import AdaptiveRetriever

            if self._embedding_store:
                self._adaptive_retriever = AdaptiveRetriever(
                    store=self._embedding_store,
                    logger=self._logger,
                )
        except ImportError:
            self._adaptive_retriever = None

    @property
    def tracker(self) -> IndexTracker:
        """Get the index tracker."""
        return self._tracker

    @property
    def embedding_store(self):
        """Get the multi-embedding store."""
        return self._embedding_store

    @property
    def logger(self):
        """Get the retrieval logger."""
        return self._logger

    @property
    def llm_provider(self):
        """Get the LLM provider."""
        return self._llm_provider

    @property
    def adaptive_retriever(self):
        """Get the adaptive retriever."""
        return self._adaptive_retriever

    def _get_rag(self):
        """Lazy load the LightRAG instance."""
        if self._rag is None:
            try:
                from lightrag import LightRAG
                from lightrag.base import EmbeddingFunc
                from lightrag.llm.openai import openai_complete_if_cache, openai_embed
                from functools import partial

                async def llm_func(prompt, **kwargs):
                    return await openai_complete_if_cache(
                        self.model_name,
                        prompt,
                        **kwargs,
                    )

                # Embedding dimension for text-embedding-3-small is 1536
                embedding_dim = 1536
                if "large" in self.embedding_model:
                    embedding_dim = 3072

                # Create proper EmbeddingFunc wrapper
                embedding_func = EmbeddingFunc(
                    embedding_dim=embedding_dim,
                    func=partial(openai_embed.func, model=self.embedding_model),
                    model_name=self.embedding_model,
                )

                self._rag = LightRAG(
                    working_dir=str(self.working_dir),
                    llm_model_func=llm_func,
                    llm_model_name=self.model_name,
                    embedding_func=embedding_func,
                )
            except ImportError:
                raise ImportError(
                    "LightRAG not installed. Run: pip install lightrag-hku"
                )
        return self._rag

    async def _ensure_initialized(self):
        """Ensure LightRAG storages are initialized."""
        rag = self._get_rag()
        if not self._initialized:
            await rag.initialize_storages()
            self._initialized = True
        return rag

    async def index_documents(
        self,
        documents: list[Document],
        batch_size: int = 10,
        skip_tracked: bool = True,
        index_to_multi_embedding: bool = True,
    ) -> dict:
        """
        Index documents into LightRAG and optionally multi-embedding store.

        Args:
            documents: List of documents to index.
            batch_size: Number of documents to process in each batch.
            skip_tracked: Skip files that are already indexed (default: True).
            index_to_multi_embedding: Also index to multi-embedding store.

        Returns:
            Indexing statistics.
        """
        rag = await self._ensure_initialized()
        indexed = 0
        skipped = 0
        errors = []
        indexed_files = set()

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            for doc in batch:
                try:
                    # Check if file is already indexed
                    if skip_tracked and doc.source_path:
                        file_path = Path(doc.source_path)
                        if self._tracker.is_indexed(file_path):
                            skipped += 1
                            continue

                    # Prepare text with metadata context
                    text_with_context = self._prepare_text_for_indexing(doc)
                    await rag.ainsert(text_with_context)

                    # Also index to multi-embedding store
                    if index_to_multi_embedding and self._embedding_store:
                        await self._embedding_store.add_document(
                            doc_id=doc.doc_id,
                            content=doc.text,
                            metadata={
                                "source_name": doc.source_name,
                                "source_type": (
                                    doc.source_type.value if doc.source_type else None
                                ),
                                "source_path": doc.source_path,
                            },
                        )

                    indexed += 1

                    # Track the indexed file
                    if doc.source_path:
                        indexed_files.add(doc.source_path)

                except Exception as e:
                    errors.append(
                        {
                            "doc_id": doc.doc_id,
                            "error": str(e),
                        }
                    )

        # Update tracker for newly indexed files
        for file_path in indexed_files:
            self._tracker.mark_indexed(Path(file_path))

        self._indexed_count += indexed

        return {
            "indexed": indexed,
            "skipped": skipped,
            "errors": len(errors),
            "error_details": errors[:10],  # Limit error details
            "total_indexed": self._indexed_count,
            "multi_embedding_enabled": self._embedding_store is not None,
        }

    def index_documents_sync(
        self,
        documents: list[Document],
        batch_size: int = 10,
    ) -> dict:
        """Synchronous version of index_documents."""
        return asyncio.run(self.index_documents(documents, batch_size))

    async def query(
        self,
        question: str,
        mode: QueryMode = QueryMode.HYBRID,
        embedding_slot: Optional[str] = None,
    ) -> QueryResult:
        """
        Query the RAG system with observability logging.

        Args:
            question: The question to ask.
            mode: Query mode (naive, local, global, hybrid, adaptive).
            embedding_slot: Specific embedding slot to use (for multi-embedding).

        Returns:
            QueryResult with answer and metadata.
        """
        # Start observability logging
        request_id = ""
        if self._logger:
            request_id = self._logger.log_query_start(
                query=question,
                query_type=mode.value,
                metadata={"embedding_slot": embedding_slot},
            )

        start_time = time.time()

        try:
            # Handle adaptive mode
            if mode == QueryMode.ADAPTIVE and self._adaptive_retriever:
                return await self._query_adaptive(question, request_id)

            rag = await self._ensure_initialized()
            from lightrag import QueryParam

            # Log embedding stage
            embedding_start = time.time()

            # First, get the context to extract sources
            context_result = await rag.aquery(
                question,
                param=QueryParam(mode=mode.value, only_need_context=True),
            )

            embedding_time = (time.time() - embedding_start) * 1000
            if self._logger and request_id:
                self._logger.log_embedding(
                    request_id=request_id,
                    slot_name=embedding_slot or "default",
                    model=self.embedding_model,
                    dimension=1536,
                    time_ms=embedding_time,
                    tokens_used=0,  # Not tracked by LightRAG
                    cost_usd=0.0,
                )

            # Extract sources from context
            sources = self._extract_sources_from_context(context_result)

            # Log search stage
            if self._logger and request_id:
                self._logger.log_search(
                    request_id=request_id,
                    strategy=mode.value,
                    slots_used=[embedding_slot or "default"],
                    top_k=5,
                    candidates=len(sources),
                    time_ms=0,
                    vector_scores=[s.get("relevance_score", 0) for s in sources],
                    final_scores=[s.get("relevance_score", 0) for s in sources],
                )

            # Build source list for citation instructions
            source_list = []
            for idx, src in enumerate(sources, 1):
                page_info = f", p.{src['page']}" if src.get("page") else ""
                source_list.append(f"[†{idx}] {src['file_name']}{page_info}")

            # Custom prompt to include inline citations
            citation_prompt = self._build_citation_prompt(source_list)

            # Generate response
            response_start = time.time()

            # Use LLM provider if available, otherwise fall back to LightRAG
            if self._llm_provider:
                result = await self._generate_with_provider(
                    question, context_result, citation_prompt
                )
            else:
                result = await rag.aquery(
                    question,
                    param=QueryParam(mode=mode.value, user_prompt=citation_prompt),
                )

            response_time = (time.time() - response_start) * 1000

            # Log response stage
            if self._logger and request_id:
                self._logger.log_response(
                    request_id=request_id,
                    llm_model=self.model_name,
                    context_tokens=len(context_result) // 4 if context_result else 0,
                    response_tokens=len(result) // 4 if result else 0,
                    time_ms=response_time,
                    cost_usd=0.0,  # Could be calculated from provider
                )

            # Enhance answer with properly formatted references
            enhanced_answer = self._enhance_references(result, sources)

            # Finalize logging
            if self._logger and request_id:
                self._logger.log_selection(
                    request_id=request_id,
                    doc_ids=[s.get("file_name", "") for s in sources],
                    reason="relevance_score",
                )
                self._logger.finalize(request_id)

            total_time = (time.time() - start_time) * 1000

            return QueryResult(
                answer=enhanced_answer,
                mode=mode,
                sources=sources,
                metadata={
                    "query_mode": mode.value,
                    "working_dir": str(self.working_dir),
                    "context_length": len(context_result) if context_result else 0,
                    "request_id": request_id,
                    "total_time_ms": round(total_time, 2),
                    "embedding_slot": embedding_slot,
                    "observability_enabled": self._logger is not None,
                },
            )

        except Exception as e:
            # Log error and finalize
            if self._logger and request_id:
                self._logger.finalize(request_id)

            return QueryResult(
                answer=f"Error processing query: {str(e)}",
                mode=mode,
                sources=[],
                metadata={"error": str(e), "request_id": request_id},
            )

    async def _query_adaptive(
        self,
        question: str,
        request_id: str,
    ) -> QueryResult:
        """Execute adaptive query using AdaptiveRetriever."""
        if not self._adaptive_retriever:
            # Fall back to hybrid mode
            return await self.query(question, QueryMode.HYBRID)

        # Classify query and get strategy
        query_type, confidence = await self._adaptive_retriever.classify_query(question)
        strategy = self._adaptive_retriever.select_strategy(query_type)

        # Perform retrieval using multi-embedding store
        retrieval_result = await self._adaptive_retriever.retrieve(
            query=question,
            request_id=request_id,
        )

        # Get documents for context building
        # For now, fall back to LightRAG for actual generation
        # since we need the knowledge graph context
        rag = await self._ensure_initialized()
        from lightrag import QueryParam

        # Determine LightRAG mode based on strategy
        lightrag_mode = "hybrid"  # Default
        if strategy.search_type == "dense":
            lightrag_mode = "naive"

        context_result = await rag.aquery(
            question,
            param=QueryParam(mode=lightrag_mode, only_need_context=True),
        )

        sources = self._extract_sources_from_context(context_result)

        source_list = []
        for idx, src in enumerate(sources, 1):
            page_info = f", p.{src['page']}" if src.get("page") else ""
            source_list.append(f"[†{idx}] {src['file_name']}{page_info}")

        citation_prompt = self._build_citation_prompt(source_list)

        if self._llm_provider:
            result = await self._generate_with_provider(
                question, context_result, citation_prompt
            )
        else:
            result = await rag.aquery(
                question,
                param=QueryParam(mode=lightrag_mode, user_prompt=citation_prompt),
            )

        enhanced_answer = self._enhance_references(result, sources)

        if self._logger and request_id:
            self._logger.finalize(request_id)

        return QueryResult(
            answer=enhanced_answer,
            mode=QueryMode.ADAPTIVE,
            sources=sources,
            metadata={
                "query_mode": "adaptive",
                "detected_query_type": query_type.value,
                "classification_confidence": confidence,
                "strategy_used": strategy.name,
                "slots_used": retrieval_result.slots_used,
                "working_dir": str(self.working_dir),
                "request_id": request_id,
            },
        )

    async def _generate_with_provider(
        self,
        question: str,
        context: str,
        citation_prompt: str,
    ) -> str:
        """Generate response using LLM provider."""
        from .llm import Message

        messages = [
            Message(role="system", content=citation_prompt),
            Message(
                role="user",
                content=f"Context:\n{context}\n\nQuestion: {question}",
            ),
        ]

        result = await self._llm_provider.complete(
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )

        return result.content

    def _build_citation_prompt(self, source_list: list[str]) -> str:
        """Build citation prompt for LLM."""
        return f"""당신은 정확한 인용을 하는 전문 리서처입니다. 아래 규칙을 엄격히 따르세요.

## 핵심 원칙 (가장 중요!)
**오직 아래 제공된 출처에 명시적으로 적힌 내용만 답변에 사용하세요.**
- 출처에 없는 내용을 추측하거나 일반 지식으로 보충하지 마세요.
- 출처 내용을 과장하거나 확대 해석하지 마세요.
- 확실하지 않으면 "해당 정보는 제공된 출처에서 확인되지 않습니다"라고 답하세요.

## 인라인 인용 규칙 (필수!)
**모든 문장 끝에 반드시 출처 번호 [†1], [†2] 형식을 표기하세요.**
- 각 문장이 끝날 때마다 해당 정보의 출처를 표시해야 합니다.
- 인용 번호는 마침표 앞에 붙입니다.
- 여러 출처가 같은 내용을 다루면 [†1][†2]처럼 병기합니다.

### 올바른 예시:
✅ "미국은 2008년 금융위기 이후 리쇼어링 정책을 시행하였습니다[†1]."
✅ "트럼프 정부는 관세를 통한 제조업 재건을 강조하였습니다[†1]. 바이든 정부는 IRA를 통해 세금 혜택을 제공하고 있습니다[†2]."
✅ "제조업 고용 비중은 감소하고 있습니다[†1][†2]."

### 잘못된 예시:
❌ "미국은 리쇼어링 정책을 시행하였습니다." (인용 번호 누락)
❌ 문단 끝에만 인용 표시 (각 문장마다 필요)

## 답변 형식
- 한국어로 작성
- 번호 목록이나 소제목으로 구조화
- "### References" 섹션은 추가하지 마세요 (자동 생성됨)

## 제공된 출처 (이 내용만 사용하세요!):
{chr(10).join(source_list)}

---
질문에 답변할 때, 위 출처에 명시된 내용만 사용하고 **반드시 모든 문장 끝에 [†번호]를 표기**하세요."""

    def _extract_sources_from_context(self, context: str) -> list[dict]:
        """Extract source information from LightRAG context with relevance scores."""
        if not context:
            return []

        sources = []
        source_counts = {}

        source_pattern = re.compile(r"\[Source:\s*([^\]]+)\]")
        page_pattern = re.compile(r"\[Page\s*(\d+)\]")

        source_matches = list(source_pattern.finditer(context))
        total_matches = len(source_matches)

        if total_matches == 0:
            return []

        for idx, match in enumerate(source_matches):
            file_name = match.group(1).strip()
            source_counts[file_name] = source_counts.get(file_name, 0) + 1

            start_pos = match.end()
            if idx + 1 < total_matches:
                end_pos = source_matches[idx + 1].start()
            else:
                end_pos = min(start_pos + 2000, len(context))

            chunk = context[start_pos:end_pos]
            page_match = page_pattern.search(chunk)
            page_num = int(page_match.group(1)) if page_match else None

            excerpt = chunk
            excerpt = re.sub(r"\[Type:[^\]]+\]", "", excerpt)
            excerpt = re.sub(r"\[Page\s*\d+\]", "", excerpt)
            excerpt = re.sub(r"-----+", "", excerpt)
            excerpt = re.sub(r"```json.*?```", "", excerpt, flags=re.DOTALL)
            excerpt = re.sub(r'\{"entity":[^}]+\}', "", excerpt)
            excerpt = excerpt.strip()

            if len(excerpt) < 20:
                continue

            position_score = 1.0 - (idx / max(total_matches, 1)) * 0.6
            content_score = min(len(excerpt) / 500, 1.0)
            base_score = (position_score * 0.4) + (content_score * 0.4)

            if len(excerpt) > 400:
                excerpt = excerpt[:400] + "..."

            sources.append(
                {
                    "file_name": file_name,
                    "page": page_num,
                    "excerpt": excerpt,
                    "relevance_score": round(base_score, 3),
                }
            )

        max_count = max(source_counts.values()) if source_counts else 1
        for source in sources:
            if source["file_name"]:
                freq_bonus = (
                    source_counts.get(source["file_name"], 1) / max_count
                ) * 0.2
                source["relevance_score"] = round(
                    min(source["relevance_score"] + freq_bonus, 1.0), 3
                )

        sources.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        seen_sources = set()
        unique_sources = []
        for source in sources:
            source_key = f"{source['file_name']}:{source.get('page', 'N/A')}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_sources.append(source)

        min_relevance = 0.4
        filtered_sources = [
            s for s in unique_sources if s.get("relevance_score", 0) >= min_relevance
        ]

        return filtered_sources[:5]

    def _enhance_references(self, answer: str, sources: list[dict]) -> str:
        """Enhance answer with properly formatted references."""
        if not sources:
            return answer

        references_lines = ["\n\n### References"]
        seen_refs = set()

        for idx, source in enumerate(sources, 1):
            file_name = source.get("file_name", "Unknown")
            page = source.get("page")

            if page:
                ref_text = f"- [†{idx}] {file_name}, p.{page}"
            else:
                ref_text = f"- [†{idx}] {file_name}"

            ref_key = f"{file_name}:{page}"
            if ref_key not in seen_refs:
                seen_refs.add(ref_key)
                references_lines.append(ref_text)

        new_references = "\n".join(references_lines)

        patterns = [
            r"\n*###\s*References\s*\n[\s\S]*$",
            r"\n*##\s*References\s*\n[\s\S]*$",
            r"\n*References:?\s*\n[\s\S]*$",
        ]

        cleaned_answer = answer
        for pattern in patterns:
            cleaned_answer = re.sub(pattern, "", cleaned_answer, flags=re.IGNORECASE)

        return cleaned_answer.rstrip() + new_references

    def query_sync(
        self,
        question: str,
        mode: QueryMode = QueryMode.HYBRID,
    ) -> QueryResult:
        """Synchronous version of query."""
        return asyncio.run(self.query(question, mode))

    def _prepare_text_for_indexing(self, doc: Document) -> str:
        """Prepare document text for indexing with metadata context."""
        parts = []

        if doc.source_name:
            parts.append(f"[Source: {doc.source_name}]")
        if doc.source_type:
            parts.append(f"[Type: {doc.source_type.value}]")

        parts.append(doc.text)

        return "\n".join(parts)

    def get_stats(self) -> dict:
        """Get comprehensive engine statistics."""
        tracker_stats = self._tracker.get_stats()

        stats = {
            "working_dir": str(self.working_dir),
            "indexed_documents": self._indexed_count,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "is_initialized": self._rag is not None,
            "tracked_files": tracker_stats["total_indexed_files"],
            "tracked_size_bytes": tracker_stats["total_size_bytes"],
            # OKT-RAG features
            "okt_rag": {
                "observability_enabled": self._logger is not None,
                "multi_embedding_enabled": self._embedding_store is not None,
                "llm_provider": (
                    self._llm_provider.model_name if self._llm_provider else None
                ),
                "adaptive_retrieval_enabled": self._adaptive_retriever is not None,
            },
        }

        # Add multi-embedding stats
        if self._embedding_store:
            stats["embedding_slots"] = self._embedding_store.get_stats()

        # Add observability stats
        if self._logger:
            try:
                stats["retrieval_stats"] = self._logger.get_stats()
            except Exception:
                stats["retrieval_stats"] = None

        # Add adaptive retrieval info
        if self._adaptive_retriever:
            stats["retrieval_strategies"] = self._adaptive_retriever.get_strategy_info()

        return stats

    def clear_index(self) -> bool:
        """Clear all indexed data."""
        import shutil

        try:
            if self.working_dir.exists():
                shutil.rmtree(self.working_dir)
                self.working_dir.mkdir(parents=True, exist_ok=True)
            self._rag = None
            self._indexed_count = 0
            self._tracker = IndexTracker(self.working_dir)

            # Clear multi-embedding store
            if self._embedding_store:
                self._embedding_store.clear()

            return True
        except Exception:
            return False

    def get_indexed_files(self) -> list[dict]:
        """Get list of all indexed files."""
        files = self._tracker.get_indexed_files()
        return [
            {
                "file_name": f.file_name,
                "file_path": f.file_path,
                "file_size": f.file_size,
                "indexed_at": f.indexed_at,
                "doc_count": f.doc_count,
            }
            for f in files
        ]

    def is_file_indexed(self, file_path: str | Path) -> bool:
        """Check if a specific file is already indexed."""
        return self._tracker.is_indexed(Path(file_path))

    async def verify_citations(
        self,
        answer: str,
        sources: list[dict],
    ) -> VerificationResult:
        """Verify that citations accurately reflect source content."""
        citation_pattern = re.compile(
            r"([^.!?\n]*[가-힣a-zA-Z]{2,}[^.!?\n]*(?:\[†?\d+\])+[.!?]?)"
        )
        matches = citation_pattern.findall(answer)

        verifications = []
        accurate_count = 0
        inaccurate_count = 0
        uncertain_count = 0

        verified_combinations = set()

        for match in matches:
            citation_nums = re.findall(r"\[†?(\d+)\]", match)
            statement = re.sub(r"\[†?\d+\]", "", match).strip()

            if not statement or len(statement) < 5 or not citation_nums:
                continue
            if not re.search(r"[가-힣a-zA-Z]{2,}", statement):
                continue

            for num_str in citation_nums:
                num = int(num_str)
                if num < 1 or num > len(sources):
                    continue

                combo_key = f"{statement[:50]}:{num}"
                if combo_key in verified_combinations:
                    continue
                verified_combinations.add(combo_key)

                source = sources[num - 1]
                source_file = source.get("file_name", "Unknown")
                source_page = source.get("page")
                source_excerpt = source.get("excerpt", "")

                if not source_excerpt or len(source_excerpt) < 20:
                    continue

                verification = await self._verify_single_citation(
                    statement=statement,
                    source_excerpt=source_excerpt,
                    source_file=source_file,
                    citation_number=num,
                    source_page=source_page,
                )
                verifications.append(verification)

                if verification.confidence >= 0.7:
                    if verification.is_accurate:
                        accurate_count += 1
                    else:
                        inaccurate_count += 1
                else:
                    uncertain_count += 1

        total = len(verifications)
        accuracy_rate = accurate_count / total if total > 0 else 0.0

        return VerificationResult(
            total_citations=total,
            verified_count=total,
            accurate_count=accurate_count,
            inaccurate_count=inaccurate_count,
            uncertain_count=uncertain_count,
            accuracy_rate=accuracy_rate,
            verifications=verifications,
        )

    async def _verify_single_citation(
        self,
        statement: str,
        source_excerpt: str,
        source_file: str,
        citation_number: int,
        source_page: Optional[int],
    ) -> CitationVerification:
        """Verify a single citation using LLM provider."""
        prompt = f"""다음 문장이 출처 내용을 정확하게 인용했는지 검증해주세요.

## 검증할 문장:
"{statement}"

## 출처 내용 (발췌):
"{source_excerpt[:1000]}"

## 검증 기준:
1. 문장의 핵심 주장이 출처 내용에서 뒷받침되는가?
2. 사실 관계가 왜곡되거나 과장되지 않았는가?
3. 출처에 없는 내용을 추가하지 않았는가?

## 응답 형식 (JSON):
{{
    "is_accurate": true/false,
    "confidence": 0.0-1.0,
    "explanation": "검증 결과 설명 (한국어, 1-2문장)"
}}

JSON만 응답하세요."""

        try:
            # Use LLM provider if available
            if self._llm_provider:
                from .llm import Message

                result = await self._llm_provider.complete(
                    messages=[Message(role="user", content=prompt)],
                    temperature=0.1,
                    max_tokens=200,
                )
                result_text = result.content.strip()
            else:
                # Fall back to direct openai call
                import openai

                client = openai.AsyncOpenAI()
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200,
                )
                result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            result_text = re.sub(r"^```json\s*", "", result_text)
            result_text = re.sub(r"\s*```$", "", result_text)
            result = json.loads(result_text)

            return CitationVerification(
                citation_number=citation_number,
                statement=statement,
                source_file=source_file,
                source_page=source_page,
                source_excerpt=(
                    source_excerpt[:300] + "..."
                    if len(source_excerpt) > 300
                    else source_excerpt
                ),
                is_accurate=result.get("is_accurate", False),
                confidence=float(result.get("confidence", 0.5)),
                explanation=result.get("explanation", "검증 실패"),
            )
        except Exception as e:
            return CitationVerification(
                citation_number=citation_number,
                statement=statement,
                source_file=source_file,
                source_page=source_page,
                source_excerpt=(
                    source_excerpt[:300] + "..."
                    if len(source_excerpt) > 300
                    else source_excerpt
                ),
                is_accurate=False,
                confidence=0.0,
                explanation=f"검증 중 오류 발생: {str(e)}",
            )

    def verify_citations_sync(
        self,
        answer: str,
        sources: list[dict],
    ) -> VerificationResult:
        """Synchronous version of verify_citations."""
        return asyncio.run(self.verify_citations(answer, sources))

    async def query_with_verification(
        self,
        question: str,
        mode: QueryMode = QueryMode.HYBRID,
        confidence_threshold: float = 0.7,
    ) -> VerifiedQueryResult:
        """Query with automatic citation verification."""
        # Step 1: Run normal query
        query_result = await self.query(question, mode)
        original_answer = query_result.answer
        sources = query_result.sources

        # Log verification start
        request_id = query_result.metadata.get("request_id", "")

        # Step 2: Verify citations
        verification_result = await self.verify_citations(original_answer, sources)

        # Log verification results
        if self._logger and request_id:
            self._logger.log_verification(
                request_id=request_id,
                citation_count=verification_result.total_citations,
                accuracy=verification_result.accuracy_rate,
                removed=[
                    v.citation_number
                    for v in verification_result.verifications
                    if not v.is_accurate and v.confidence >= confidence_threshold
                ],
            )

        # Step 3: Identify inaccurate citations
        inaccurate_citations = set()
        verification_log = []

        for v in verification_result.verifications:
            log_entry = {
                "citation_number": v.citation_number,
                "statement": (
                    v.statement[:100] + "..." if len(v.statement) > 100 else v.statement
                ),
                "source_file": v.source_file,
                "source_page": v.source_page,
                "is_accurate": v.is_accurate,
                "confidence": v.confidence,
                "explanation": v.explanation,
                "status": (
                    "✅ 정확"
                    if (v.is_accurate and v.confidence >= confidence_threshold)
                    else (
                        "❌ 부정확"
                        if (not v.is_accurate and v.confidence >= confidence_threshold)
                        else "⚠️ 불확실"
                    )
                ),
            }
            verification_log.append(log_entry)

            if v.confidence >= confidence_threshold and not v.is_accurate:
                inaccurate_citations.add(v.citation_number)

        # Step 4: Build citation renumbering map
        old_to_new = {}
        new_num = 1
        for old_num in range(1, len(sources) + 1):
            if old_num not in inaccurate_citations:
                old_to_new[old_num] = new_num
                new_num += 1

        # Step 5: Replace citation numbers
        corrected_answer = original_answer

        for old_num in range(len(sources), 0, -1):
            if old_num in inaccurate_citations:
                corrected_answer = re.sub(rf"\[†?{old_num}\]", "", corrected_answer)
            else:
                corrected_answer = re.sub(
                    rf"\[†?{old_num}\]", f"[[CITE_{old_num}]]", corrected_answer
                )

        for old_num, new_num in old_to_new.items():
            corrected_answer = corrected_answer.replace(
                f"[[CITE_{old_num}]]", f"[†{new_num}]"
            )

        corrected_answer = re.sub(r"[ \t]+", " ", corrected_answer)
        corrected_answer = re.sub(r" +([.!?,])", r"\1", corrected_answer)
        corrected_answer = re.sub(r"\n{3,}", "\n\n", corrected_answer)
        corrected_answer = re.sub(r" +\n", "\n", corrected_answer)

        # Step 6: Update References section
        corrected_answer = self._update_references_section(
            corrected_answer, sources, inaccurate_citations, old_to_new
        )

        return VerifiedQueryResult(
            original_answer=original_answer,
            corrected_answer=corrected_answer,
            mode=mode,
            sources=sources,
            verification_log=verification_log,
            accuracy_rate=verification_result.accuracy_rate,
            removed_citations=list(inaccurate_citations),
            metadata={
                **query_result.metadata,
                "total_citations": verification_result.total_citations,
                "accurate_count": verification_result.accurate_count,
                "inaccurate_count": verification_result.inaccurate_count,
                "uncertain_count": verification_result.uncertain_count,
            },
        )

    def _update_references_section(
        self,
        answer: str,
        sources: list[dict],
        removed_citations: set[int],
        old_to_new: dict[int, int] = None,
    ) -> str:
        """Update References section with renumbered citations."""
        patterns = [
            r"\n*###\s*References\s*\n[\s\S]*$",
            r"\n*##\s*References\s*\n[\s\S]*$",
            r"\n*References:?\s*\n[\s\S]*$",
        ]

        cleaned_answer = answer
        for pattern in patterns:
            cleaned_answer = re.sub(pattern, "", cleaned_answer, flags=re.IGNORECASE)

        references_lines = ["\n\n### References"]
        seen_refs = set()

        for old_idx, source in enumerate(sources, 1):
            if old_idx in removed_citations:
                continue

            if old_to_new and old_idx in old_to_new:
                new_idx = old_to_new[old_idx]
            else:
                new_idx = old_idx

            file_name = source.get("file_name", "Unknown")
            page = source.get("page")

            ref_key = f"{file_name}:{page}"
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)

            if page:
                references_lines.append(f"- [†{new_idx}] {file_name}, p.{page}")
            else:
                references_lines.append(f"- [†{new_idx}] {file_name}")

        if len(references_lines) > 1:
            return cleaned_answer.rstrip() + "\n".join(references_lines)
        return cleaned_answer.rstrip()

    def query_with_verification_sync(
        self,
        question: str,
        mode: QueryMode = QueryMode.HYBRID,
    ) -> VerifiedQueryResult:
        """Synchronous version of query_with_verification."""
        return asyncio.run(self.query_with_verification(question, mode))
