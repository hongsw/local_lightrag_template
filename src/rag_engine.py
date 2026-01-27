"""
LightRAG engine wrapper for knowledge graph + vector RAG.
"""

import asyncio
import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from .connectors.base import Document
from .index_tracker import IndexTracker


class QueryMode(str, Enum):
    """LightRAG query modes."""
    NAIVE = "naive"       # Simple vector similarity
    LOCAL = "local"       # Local context from knowledge graph
    GLOBAL = "global"     # Global patterns from knowledge graph
    HYBRID = "hybrid"     # Combines local and global


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
    LightRAG engine wrapper providing knowledge graph + vector RAG capabilities.

    LightRAG combines:
    - Vector store for semantic similarity
    - Knowledge graph for entity relationships
    - Entity store for structured information
    """

    def __init__(
        self,
        working_dir: str | Path,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize the RAG engine.

        Args:
            working_dir: Directory for LightRAG data storage.
            openai_api_key: OpenAI API key (uses env var if not provided).
            model_name: LLM model for generation.
            embedding_model: Model for text embeddings.
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

    @property
    def tracker(self) -> IndexTracker:
        """Get the index tracker."""
        return self._tracker

    def _get_rag(self):
        """Lazy load the LightRAG instance."""
        if self._rag is None:
            try:
                from lightrag import LightRAG
                from lightrag.base import EmbeddingFunc
                from lightrag.llm.openai import openai_complete_if_cache, openai_embed
                from functools import partial
                import numpy as np

                async def llm_func(prompt, **kwargs):
                    return await openai_complete_if_cache(
                        self.model_name,
                        prompt,
                        **kwargs
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
    ) -> dict:
        """
        Index documents into LightRAG.

        Args:
            documents: List of documents to index.
            batch_size: Number of documents to process in each batch.
            skip_tracked: Skip files that are already indexed (default: True).

        Returns:
            Indexing statistics.
        """
        rag = await self._ensure_initialized()
        indexed = 0
        skipped = 0
        errors = []
        indexed_files = set()

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

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
                    indexed += 1

                    # Track the indexed file
                    if doc.source_path:
                        indexed_files.add(doc.source_path)

                except Exception as e:
                    errors.append({
                        "doc_id": doc.doc_id,
                        "error": str(e),
                    })

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
    ) -> QueryResult:
        """
        Query the RAG system.

        Args:
            question: The question to ask.
            mode: Query mode (naive, local, global, hybrid).

        Returns:
            QueryResult with answer and metadata.
        """
        rag = await self._ensure_initialized()

        try:
            from lightrag import QueryParam

            # First, get the context to extract sources
            context_result = await rag.aquery(
                question,
                param=QueryParam(mode=mode.value, only_need_context=True)
            )

            # Extract sources from context
            sources = self._extract_sources_from_context(context_result)

            # Build source list for citation instructions
            source_list = []
            for idx, src in enumerate(sources, 1):
                page_info = f", p.{src['page']}" if src.get('page') else ""
                source_list.append(f"[†{idx}] {src['file_name']}{page_info}")

            # Custom prompt to include inline citations
            citation_prompt = f"""당신은 정확한 인용을 하는 전문 리서처입니다. 아래 규칙을 엄격히 따르세요.

## 핵심 원칙 (가장 중요!)
**오직 아래 제공된 출처에 명시적으로 적힌 내용만 답변에 사용하세요.**
- 출처에 없는 내용을 추측하거나 일반 지식으로 보충하지 마세요.
- 출처 내용을 과장하거나 확대 해석하지 마세요.
- 확실하지 않으면 "해당 정보는 제공된 출처에서 확인되지 않습니다"라고 답하세요.

## 인용 규칙
1. 각 문장/주장 끝에 해당 출처 번호를 [†1], [†2] 형식으로 표기하세요.
2. 인용 번호는 해당 내용이 실제로 있는 출처만 사용하세요.
3. 여러 출처가 같은 내용을 다루면 [†1][†2]처럼 병기하세요.

## 답변 형식
- 한국어로 작성
- 번호 목록이나 소제목으로 구조화
- "### References" 섹션은 추가하지 마세요 (자동 생성됨)

## 제공된 출처 (이 내용만 사용하세요!):
{chr(10).join(source_list)}

---
질문에 답변할 때, 위 출처에 명시된 내용만 사용하고 각 주장에 [†번호]를 표기하세요.
"""

            # Then get the actual answer with citation instructions
            result = await rag.aquery(
                question,
                param=QueryParam(mode=mode.value, user_prompt=citation_prompt)
            )

            # Enhance answer with properly formatted references including page numbers
            enhanced_answer = self._enhance_references(result, sources)

            return QueryResult(
                answer=enhanced_answer,
                mode=mode,
                sources=sources,
                metadata={
                    "query_mode": mode.value,
                    "working_dir": str(self.working_dir),
                    "context_length": len(context_result) if context_result else 0,
                },
            )
        except Exception as e:
            return QueryResult(
                answer=f"Error processing query: {str(e)}",
                mode=mode,
                sources=[],
                metadata={"error": str(e)},
            )

    def _extract_sources_from_context(self, context: str) -> list[dict]:
        """
        Extract source information from LightRAG context with relevance scores.

        Parses the context to find file names, page numbers, and relevant excerpts.
        Calculates relevance scores based on:
        - Position in context (earlier = higher relevance)
        - Content length (more content = higher relevance)
        - Frequency of source mentions
        """
        if not context:
            return []

        sources = []
        source_counts = {}  # Track how many times each source appears

        # Find all [Source: ...] markers with their positions
        source_pattern = re.compile(r'\[Source:\s*([^\]]+)\]')
        page_pattern = re.compile(r'\[Page\s*(\d+)\]')

        # Find all source markers
        source_matches = list(source_pattern.finditer(context))
        total_matches = len(source_matches)

        if total_matches == 0:
            return []

        for idx, match in enumerate(source_matches):
            file_name = match.group(1).strip()

            # Count source frequency
            source_counts[file_name] = source_counts.get(file_name, 0) + 1

            # Find the content after this source marker until the next source marker
            start_pos = match.end()
            if idx + 1 < total_matches:
                end_pos = source_matches[idx + 1].start()
            else:
                end_pos = min(start_pos + 2000, len(context))  # Limit to 2000 chars

            chunk = context[start_pos:end_pos]

            # Extract page number from the chunk
            page_match = page_pattern.search(chunk)
            page_num = int(page_match.group(1)) if page_match else None

            # Clean excerpt - remove metadata tags
            excerpt = chunk
            excerpt = re.sub(r'\[Type:[^\]]+\]', '', excerpt)
            excerpt = re.sub(r'\[Page\s*\d+\]', '', excerpt)
            excerpt = re.sub(r'-----+', '', excerpt)
            excerpt = re.sub(r'```json.*?```', '', excerpt, flags=re.DOTALL)
            excerpt = re.sub(r'\{"entity":[^}]+\}', '', excerpt)
            excerpt = excerpt.strip()

            # Skip if no meaningful excerpt
            if len(excerpt) < 20:
                continue

            # Calculate relevance score
            # Position score: earlier in context = higher relevance (0.4 weight)
            position_score = 1.0 - (idx / max(total_matches, 1)) * 0.6

            # Content score: longer meaningful content = higher relevance (0.4 weight)
            content_score = min(len(excerpt) / 500, 1.0)

            # Combined base score
            base_score = (position_score * 0.4) + (content_score * 0.4)

            # Truncate excerpt if too long
            if len(excerpt) > 400:
                excerpt = excerpt[:400] + "..."

            sources.append({
                "file_name": file_name,
                "page": page_num,
                "excerpt": excerpt,
                "relevance_score": round(base_score, 3),
            })

        # Add frequency bonus to relevance scores (0.2 weight)
        max_count = max(source_counts.values()) if source_counts else 1
        for source in sources:
            if source["file_name"]:
                freq_bonus = (source_counts.get(source["file_name"], 1) / max_count) * 0.2
                source["relevance_score"] = round(
                    min(source["relevance_score"] + freq_bonus, 1.0), 3
                )

        # Sort by relevance score (descending)
        sources.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Deduplicate: keep highest-scoring entry per file+page combination
        seen_sources = set()
        unique_sources = []
        for source in sources:
            source_key = f"{source['file_name']}:{source.get('page', 'N/A')}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_sources.append(source)

        # Filter by minimum relevance threshold and limit to top 5
        # Reduced from 10 to 5 to improve citation accuracy
        min_relevance = 0.4
        filtered_sources = [s for s in unique_sources if s.get("relevance_score", 0) >= min_relevance]

        return filtered_sources[:5]

    def _enhance_references(self, answer: str, sources: list[dict]) -> str:
        """
        Enhance the answer by replacing or adding a References section
        with properly formatted citations including page numbers.
        """
        if not sources:
            return answer

        # Build new references section with page numbers
        references_lines = ["\n\n### References"]
        seen_refs = set()

        for idx, source in enumerate(sources, 1):
            file_name = source.get("file_name", "Unknown")
            page = source.get("page")

            # Create reference with page number (using [†] format)
            if page:
                ref_text = f"- [†{idx}] {file_name}, p.{page}"
            else:
                ref_text = f"- [†{idx}] {file_name}"

            # Deduplicate references (same file + page)
            ref_key = f"{file_name}:{page}"
            if ref_key not in seen_refs:
                seen_refs.add(ref_key)
                references_lines.append(ref_text)

        new_references = "\n".join(references_lines)

        # Remove existing References section if present
        # Look for common patterns like "### References", "## References", "References:"
        patterns = [
            r'\n*###\s*References\s*\n[\s\S]*$',
            r'\n*##\s*References\s*\n[\s\S]*$',
            r'\n*References:?\s*\n[\s\S]*$',
        ]

        cleaned_answer = answer
        for pattern in patterns:
            cleaned_answer = re.sub(pattern, '', cleaned_answer, flags=re.IGNORECASE)

        # Append new references section
        return cleaned_answer.rstrip() + new_references

    def query_sync(
        self,
        question: str,
        mode: QueryMode = QueryMode.HYBRID,
    ) -> QueryResult:
        """Synchronous version of query."""
        return asyncio.run(self.query(question, mode))

    def _prepare_text_for_indexing(self, doc: Document) -> str:
        """
        Prepare document text for indexing with metadata context.

        Adds source information to help with retrieval context.
        """
        parts = []

        # Add source context
        if doc.source_name:
            parts.append(f"[Source: {doc.source_name}]")
        if doc.source_type:
            parts.append(f"[Type: {doc.source_type.value}]")

        # Add main content
        parts.append(doc.text)

        return "\n".join(parts)

    def get_stats(self) -> dict:
        """Get engine statistics."""
        tracker_stats = self._tracker.get_stats()
        return {
            "working_dir": str(self.working_dir),
            "indexed_documents": self._indexed_count,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "is_initialized": self._rag is not None,
            "tracked_files": tracker_stats["total_indexed_files"],
            "tracked_size_bytes": tracker_stats["total_size_bytes"],
        }

    def clear_index(self) -> bool:
        """
        Clear all indexed data.

        Warning: This deletes all LightRAG data in the working directory.
        """
        import shutil

        try:
            if self.working_dir.exists():
                shutil.rmtree(self.working_dir)
                self.working_dir.mkdir(parents=True, exist_ok=True)
            self._rag = None
            self._indexed_count = 0
            # Reinitialize tracker (creates empty tracker file)
            self._tracker = IndexTracker(self.working_dir)
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
        """
        Verify that citations in the answer accurately reflect the source content.

        Args:
            answer: The answer text with inline citations like [1], [2].
            sources: List of source dictionaries with file_name, page, excerpt.

        Returns:
            VerificationResult with detailed verification for each citation.
        """
        import openai

        # Extract statements with their citations from the answer
        # Pattern requires at least 2 Korean or English characters
        # Supports both [†1] and [1] formats for backward compatibility
        citation_pattern = re.compile(r'([^.!?\n]*[가-힣a-zA-Z]{2,}[^.!?\n]*(?:\[†?\d+\])+[.!?]?)')
        matches = citation_pattern.findall(answer)

        verifications = []
        accurate_count = 0
        inaccurate_count = 0
        uncertain_count = 0

        # Track already verified statement+citation combinations to avoid duplicates
        verified_combinations = set()

        for match in matches:
            # Extract citation numbers from this statement (supports [†1] and [1] formats)
            citation_nums = re.findall(r'\[†?(\d+)\]', match)
            statement = re.sub(r'\[†?\d+\]', '', match).strip()

            # Enhanced filtering: require meaningful content
            if not statement or len(statement) < 5 or not citation_nums:
                continue
            # Must contain at least 2 Korean or English characters
            if not re.search(r'[가-힣a-zA-Z]{2,}', statement):
                continue

            for num_str in citation_nums:
                num = int(num_str)
                if num < 1 or num > len(sources):
                    continue

                # Skip duplicate statement+citation combinations
                combo_key = f"{statement[:50]}:{num}"
                if combo_key in verified_combinations:
                    continue
                verified_combinations.add(combo_key)

                source = sources[num - 1]
                source_file = source.get("file_name", "Unknown")
                source_page = source.get("page")
                source_excerpt = source.get("excerpt", "")

                # Skip if source excerpt is too short or empty
                if not source_excerpt or len(source_excerpt) < 20:
                    continue

                # Use LLM to verify the citation accuracy
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
        """Verify a single citation using LLM."""
        import openai

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
            client = openai.AsyncOpenAI()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            result_text = response.choices[0].message.content.strip()
            # Parse JSON response
            import json
            # Clean up potential markdown formatting
            result_text = re.sub(r'^```json\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)
            result = json.loads(result_text)

            return CitationVerification(
                citation_number=citation_number,
                statement=statement,
                source_file=source_file,
                source_page=source_page,
                source_excerpt=source_excerpt[:300] + "..." if len(source_excerpt) > 300 else source_excerpt,
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
                source_excerpt=source_excerpt[:300] + "..." if len(source_excerpt) > 300 else source_excerpt,
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
        """
        Query with automatic citation verification.

        Runs the query, verifies all citations, removes inaccurate ones,
        and returns a corrected answer with verification log.

        Args:
            question: The question to ask.
            mode: Query mode.
            confidence_threshold: Minimum confidence to consider accurate (default: 0.7).

        Returns:
            VerifiedQueryResult with corrected answer and verification log.
        """
        # Step 1: Run normal query
        query_result = await self.query(question, mode)
        original_answer = query_result.answer
        sources = query_result.sources

        # Step 2: Verify citations
        verification_result = await self.verify_citations(original_answer, sources)

        # Step 3: Identify inaccurate citations
        inaccurate_citations = set()
        verification_log = []

        for v in verification_result.verifications:
            log_entry = {
                "citation_number": v.citation_number,
                "statement": v.statement[:100] + "..." if len(v.statement) > 100 else v.statement,
                "source_file": v.source_file,
                "source_page": v.source_page,
                "is_accurate": v.is_accurate,
                "confidence": v.confidence,
                "explanation": v.explanation,
                "status": "✅ 정확" if (v.is_accurate and v.confidence >= confidence_threshold)
                         else "❌ 부정확" if (not v.is_accurate and v.confidence >= confidence_threshold)
                         else "⚠️ 불확실"
            }
            verification_log.append(log_entry)

            # Mark as inaccurate if confidence is high and is_accurate is False
            if v.confidence >= confidence_threshold and not v.is_accurate:
                inaccurate_citations.add(v.citation_number)

        # Step 4: Build citation renumbering map (old -> new)
        old_to_new = {}
        new_num = 1
        for old_num in range(1, len(sources) + 1):
            if old_num not in inaccurate_citations:
                old_to_new[old_num] = new_num
                new_num += 1

        # Step 5: Replace citation numbers in answer (renumber remaining, remove inaccurate)
        corrected_answer = original_answer

        # First, temporarily replace all citations with placeholders (supports [†1] and [1] formats)
        for old_num in range(len(sources), 0, -1):
            if old_num in inaccurate_citations:
                # Remove inaccurate citations (both formats)
                corrected_answer = re.sub(rf'\[†?{old_num}\]', '', corrected_answer)
            else:
                # Replace with placeholder to avoid conflicts (both formats)
                corrected_answer = re.sub(rf'\[†?{old_num}\]', f'[[CITE_{old_num}]]', corrected_answer)

        # Then replace placeholders with new numbers (using [†] format)
        for old_num, new_num in old_to_new.items():
            corrected_answer = corrected_answer.replace(f'[[CITE_{old_num}]]', f'[†{new_num}]')

        # Clean up extra spaces while preserving newlines for Markdown formatting
        # Only replace multiple spaces (not newlines) with single space
        corrected_answer = re.sub(r'[ \t]+', ' ', corrected_answer)
        # Clean up spaces before punctuation
        corrected_answer = re.sub(r' +([.!?,])', r'\1', corrected_answer)
        # Clean up multiple consecutive newlines (keep max 2 for paragraph breaks)
        corrected_answer = re.sub(r'\n{3,}', '\n\n', corrected_answer)
        # Remove trailing spaces on each line
        corrected_answer = re.sub(r' +\n', '\n', corrected_answer)

        # Step 6: Update References section with renumbered sources
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
        # Find and remove existing References section
        patterns = [
            r'\n*###\s*References\s*\n[\s\S]*$',
            r'\n*##\s*References\s*\n[\s\S]*$',
            r'\n*References:?\s*\n[\s\S]*$',
        ]

        cleaned_answer = answer
        for pattern in patterns:
            cleaned_answer = re.sub(pattern, '', cleaned_answer, flags=re.IGNORECASE)

        # Build new references section with correct numbering
        references_lines = ["\n\n### References"]
        seen_refs = set()

        for old_idx, source in enumerate(sources, 1):
            if old_idx in removed_citations:
                continue

            # Get new number from mapping or calculate it
            if old_to_new and old_idx in old_to_new:
                new_idx = old_to_new[old_idx]
            else:
                new_idx = old_idx

            file_name = source.get("file_name", "Unknown")
            page = source.get("page")

            # Deduplicate references
            ref_key = f"{file_name}:{page}"
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)

            if page:
                references_lines.append(f"- [†{new_idx}] {file_name}, p.{page}")
            else:
                references_lines.append(f"- [†{new_idx}] {file_name}")

        if len(references_lines) > 1:  # Only add if there are references
            return cleaned_answer.rstrip() + "\n".join(references_lines)
        return cleaned_answer.rstrip()

    def query_with_verification_sync(
        self,
        question: str,
        mode: QueryMode = QueryMode.HYBRID,
    ) -> VerifiedQueryResult:
        """Synchronous version of query_with_verification."""
        return asyncio.run(self.query_with_verification(question, mode))
