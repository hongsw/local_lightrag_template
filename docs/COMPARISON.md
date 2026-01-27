# RAG 프레임워크 비교 분석

본 문서는 Korea Tech Education RAG 시스템과 주요 오픈소스 RAG 프레임워크를 비교 분석합니다.

## 프레임워크 개요

| 프레임워크 | 개발사 | 핵심 특징 | GitHub Stars |
|-----------|--------|----------|--------------|
| **Korea Tech RAG** | - | LightRAG + 자동 인용 검증 | Private |
| **LightRAG** | HKUDS | Knowledge Graph + Vector (경량화) | 20k+ |
| **GraphRAG** | Microsoft | Knowledge Graph + Community Detection | 25k+ |
| **LangChain** | LangChain Inc | 범용 LLM 프레임워크 | 95k+ |
| **LlamaIndex** | LlamaIndex | 데이터 인덱싱 특화 | 35k+ |
| **RAGAS** | Explodinggradients | RAG 평가 프레임워크 | 7k+ |

## 아키텍처 비교

### Vector RAG vs Graph RAG vs LightRAG

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG 아키텍처 스펙트럼                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [Vector RAG]          [LightRAG]              [GraphRAG]               │
│      │                     │                       │                    │
│  단순/빠름 ←───────────────┼───────────────→ 복잡/정확                   │
│                            │                                            │
│  • 벡터 유사도만           • 벡터 + 경량 그래프      • 전체 Knowledge Graph │
│  • 청크 독립적             • Entity 관계 추출       • Community Detection  │
│  • 빠른 검색               • Dual-level 검색       • Graph Traversal      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Korea Tech RAG 위치

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Vector RAG ──── LightRAG ──── Korea Tech RAG ──── GraphRAG            │
│                                      │                                  │
│                          ┌───────────┴───────────┐                      │
│                          │  LightRAG 기반        │                      │
│                          │  + 자동 인용 검증     │                      │
│                          │  + 웹 대시보드        │                      │
│                          │  + 프롬프트 최적화    │                      │
│                          └───────────────────────┘                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 상세 비교

### 1. 검색 방식

| 프레임워크 | 검색 방식 | 장점 | 단점 |
|-----------|----------|------|------|
| **Vector RAG** | 벡터 유사도 | 빠름, 단순 | 관계 파악 불가 |
| **LightRAG** | 벡터 + Entity Graph | 균형잡힌 성능 | 복잡한 관계 제한적 |
| **GraphRAG** | Graph Traversal | 깊은 관계 분석 | 느림, 비용 높음 |
| **Korea Tech RAG** | LightRAG Dual-level | 균형 + 검증 | LightRAG 한계 상속 |

### 2. 성능 지표

| 지표 | Vector RAG | LightRAG | GraphRAG | Korea Tech RAG |
|------|-----------|----------|----------|----------------|
| **검색 지연시간** | ~50ms | ~80ms | ~120ms+ | ~100ms |
| **인덱싱 비용** | 낮음 | 중간 | 높음 | 중간 |
| **관계 정확도** | 낮음 | 중간 | 높음 (+10%) | 중간 + 검증 |
| **증분 업데이트** | 쉬움 | 쉬움 | 어려움 | 쉬움 |

### 3. 인용/검증 기능 비교

| 기능 | LangChain | LlamaIndex | RAGAS | Korea Tech RAG |
|------|-----------|------------|-------|----------------|
| **인라인 인용** | 수동 구현 | 수동 구현 | ❌ | ✅ 자동 [†N] |
| **인용 검증** | ❌ | ❌ | Faithfulness 메트릭 | ✅ LLM 기반 검증 |
| **자동 정정** | ❌ | ❌ | ❌ | ✅ 부정확 인용 제거 |
| **검증 로그** | ❌ | ❌ | 평가 리포트 | ✅ 실시간 로그 |
| **소스 추적** | 메타데이터 | 메타데이터 | ❌ | ✅ 파일+페이지 |

### 4. 사용 편의성

| 기능 | LangChain | LlamaIndex | LightRAG | Korea Tech RAG |
|------|-----------|------------|----------|----------------|
| **웹 UI** | ❌ | ❌ | ❌ | ✅ 대시보드 |
| **REST API** | 수동 구현 | 수동 구현 | ❌ | ✅ FastAPI |
| **설정 복잡도** | 높음 | 중간 | 낮음 | 낮음 |
| **문서화** | 풍부 | 풍부 | 기본 | 상세 |

## 프레임워크별 상세 분석

### LightRAG (기반 엔진)

**출처**: [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)

**특징**:
- Graph-Enhanced Text Indexing
- Dual-Level Retrieval (Low-level + High-level)
- 벡터 기반 Entity/Relation 검색
- Community Traversal 없이 경량화

**장점**:
- GraphRAG 대비 ~30% 빠른 지연시간
- 증분 업데이트 용이
- 인덱싱 비용 절감

**단점**:
- 복잡한 인과관계 분석 제한적
- 인용 검증 기능 없음

### GraphRAG (Microsoft)

**출처**: [Microsoft GraphRAG](https://microsoft.github.io/graphrag/)

**특징**:
- 전체 Knowledge Graph 구축
- Community Detection 알고리즘
- Graph Machine Learning 활용

**장점**:
- 관계형 QA 벤치마크 ~10% 정확도 향상
- 깊은 관계 분석 가능
- 복잡한 추론 지원

**단점**:
- 인덱싱 비용 높음
- 검색 지연시간 2배
- 증분 업데이트 어려움

### RAGAS (평가 프레임워크)

**출처**: [RAGAS](https://github.com/explodinggradients/ragas)

**특징**:
- RAG 시스템 평가 전문
- Faithfulness, Relevancy 메트릭
- 자동 테스트 데이터셋 생성

**메트릭**:
- Context Precision
- Context Recall
- Faithfulness (출처 일치도)
- Answer Relevancy

**Korea Tech RAG와 차이**:
- RAGAS: 사후 평가 (배치)
- Korea Tech RAG: 실시간 검증 + 자동 정정

### LangChain / LlamaIndex

**출처**:
- [LangChain vs LlamaIndex 비교](https://latenode.com/blog/platform-comparisons-alternatives/automation-platform-comparisons/langchain-vs-llamaindex-2025-complete-rag-framework-comparison)

**특징**:
- 범용 LLM 애플리케이션 프레임워크
- 다양한 컴포넌트 조합 가능
- 풍부한 생태계

**Korea Tech RAG와 차이**:
- LangChain/LlamaIndex: 빌딩 블록 제공
- Korea Tech RAG: 완성된 솔루션 (인용 검증 포함)

## Korea Tech RAG 차별점

### 1. 자동 인용 검증 파이프라인

```
기존 RAG:  질문 → 검색 → 응답 생성 → 출력

Korea Tech RAG:
    질문 → 검색 → 응답 생성 → 인용 검증 → 정정 → 출력
                                   ↓
                            검증 로그 생성
```

### 2. 엄격한 소스 기반 프롬프트

```
"오직 제공된 출처에 명시적으로 적힌 내용만 답변에 사용하세요.
출처에 없는 내용을 추측하거나 일반 지식으로 보충하지 마세요."
```

### 3. 검증 결과 분류

| 상태 | 조건 | 처리 |
|------|------|------|
| ✅ Accurate | `is_accurate=true` & `confidence≥0.7` | 유지 |
| ❌ Inaccurate | `is_accurate=false` & `confidence≥0.7` | 제거 |
| ⚠️ Uncertain | `confidence<0.7` | 유지 (경고) |

### 4. 통합 웹 대시보드

- 실시간 쿼리/검증
- Auto-Verify Mode 토글
- 검증 로그 시각화
- 인덱싱 관리

## 사용 시나리오별 권장

| 시나리오 | 권장 프레임워크 | 이유 |
|---------|---------------|------|
| **빠른 프로토타이핑** | LangChain | 풍부한 예제, 커뮤니티 |
| **데이터 인덱싱 중심** | LlamaIndex | 다양한 데이터 소스 |
| **복잡한 관계 분석** | GraphRAG | 깊은 그래프 분석 |
| **비용 효율적 그래프 RAG** | LightRAG | 경량화된 그래프 |
| **인용 정확성 필수** | **Korea Tech RAG** | 자동 검증+정정 |
| **교육/연구 문서** | **Korea Tech RAG** | 소스 추적, 페이지 표시 |
| **RAG 품질 평가** | RAGAS | 전문 메트릭 |

## 결론

Korea Tech RAG는 LightRAG의 효율적인 Knowledge Graph RAG를 기반으로,
**자동 인용 검증**이라는 고유한 기능을 추가하여 응답의 신뢰성을 높입니다.

특히 다음 상황에 적합합니다:
- 📚 학술/연구 문서 기반 QA
- 📋 정확한 출처 표기가 필요한 리포트 생성
- ✅ 팩트체크가 중요한 도메인
- 🎓 교육 자료 기반 질의응답

## 참고 자료

- [LightRAG: Simple and Fast Alternative to GraphRAG](https://learnopencv.com/lightrag/)
- [Vector RAG vs Graph RAG vs LightRAG](https://tdg-global.net/blog/analytics/vector-rag-vs-graph-rag-vs-lightrag/kenan-agyel/)
- [GraphRAG vs LightRAG 비교 분석](https://www.maargasystems.com/2025/05/12/understanding-graphrag-vs-lightrag-a-comparative-analysis-for-enhanced-knowledge-retrieval/)
- [Microsoft GraphRAG](https://microsoft.github.io/graphrag/)
- [LangChain vs LlamaIndex 2025](https://latenode.com/blog/platform-comparisons-alternatives/automation-platform-comparisons/langchain-vs-llamaindex-2025-complete-rag-framework-comparison)
- [Best Open Source RAG Frameworks](https://www.signitysolutions.com/blog/best-open-source-rag-frameworks)
- [Best RAG Evaluation Tools](https://www.braintrust.dev/articles/best-rag-evaluation-tools)
