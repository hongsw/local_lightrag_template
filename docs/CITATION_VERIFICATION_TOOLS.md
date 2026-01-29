# Citation Verification Tools 비교 분석

LLM 기반 RAG 시스템의 인용 검증(Citation Verification) 전용 도구 및 프레임워크 비교 분석 문서

> **Last Updated**: 2025-01-29

## 개요

RAG(Retrieval-Augmented Generation) 시스템에서 **Citation Hallucination**(인용 환각)은 핵심 문제입니다. 모델이 실제 출처를 정확히 인용하지 않거나, 출처에 없는 내용을 마치 있는 것처럼 인용하는 현상이 발생합니다.

본 문서는 이러한 문제를 해결하기 위한 전용 도구들을 비교 분석합니다.

---

## 도구 개요 (10개 비교)

| 도구 | 개발사/저자 | 핵심 특징 | GitHub Stars | 라이선스 |
|-----|-----------|----------|--------------|---------|
| **OKT-RAG** | - | LightRAG + 실시간 LLM 인용 검증 + 자동 정정 | Private | MIT |
| **Citation-Check-Skill** | SerenaTian | Two-pass 검증, 7가지 상태 분류 | 30+ | MIT |
| **HaluGate** | - | Token-level 환각 탐지, ~125ms/4K | - | - |
| **Ragas** | Explorium | Faithfulness/Context 메트릭, 산업 표준 | 8k+ | Apache 2.0 |
| **TruLens** | TruEra | RAG Triad 평가, 실시간 피드백 | 2.5k+ | MIT |
| **DeepEval** | Confident AI | 14+ RAG 메트릭, pytest 통합 | 4k+ | Apache 2.0 |
| **Arize Phoenix** | Arize AI | RAG 추적 및 평가, 프로덕션 모니터링 | 3k+ | Apache 2.0 |
| **LLMWare** | LLMWare | evidence_check 함수, 경량 검증 | 7k+ | Apache 2.0 |
| **FACTUM** | Stanford | Attention 경로 분석, 기계적 탐지 | 연구용 | MIT |
| **SemanticCite** | - | 4-class 분류, 의미론적 검증 | 연구용 | - |

---

## 상세 분석

### 1. OKT-RAG (본 프로젝트)

**GitHub**: Private (본 프로젝트)

**핵심 기능**:
- **실시간 LLM 인용 검증**: 응답 생성 즉시 각 인용 [†N]을 검증
- **자동 정정**: 부정확한 인용 자동 제거 및 번호 재정렬
- **신뢰도 점수**: 각 인용별 confidence score (0.0~1.0)
- **검증 로그**: 상세 판정 이유 기록

**검증 프로세스**:
```
응답 생성 → 인용 추출 → LLM 검증 → 정정 → 최종 응답
                              ↓
                       검증 로그 생성
```

**상태 분류**:
| 상태 | 조건 | 처리 |
|------|------|------|
| ✅ Accurate | `is_accurate=true` & `confidence≥0.7` | 유지 |
| ❌ Inaccurate | `is_accurate=false` & `confidence≥0.7` | 제거 |
| ⚠️ Uncertain | `confidence<0.7` | 유지 (경고) |

**장점**: 실시간 검증, 자동 정정, 프로덕션 사용 가능
**단점**: LLM 호출 비용 추가

---

### 2. Citation-Check-Skill

**GitHub**: [serenakeyitan/Citation-Check-Skill](https://github.com/serenakeyitan/Citation-Check-Skill) (30 stars)

**핵심 기능**:
- **Two-Pass Architecture**: 1차 생성 → 2차 검증
- **7가지 상태 분류**: 세분화된 인용 품질 평가
- **경량 레이어**: 기존 RAG 시스템에 쉽게 통합

**상태 분류 (7가지)**:
| 상태 | 설명 |
|------|------|
| `Verified` | 인용이 출처와 정확히 일치 |
| `Numerical Error` | 숫자/통계 불일치 |
| `Unverified` | 출처에서 확인 불가 |
| `Hallucination` | 출처에 없는 내용 |
| `Misleading` | 문맥 왜곡 |
| `Citation Not Found` | 인용 번호가 존재하지 않음 |
| `Not in Source` | 출처에 해당 정보 없음 |

**아키텍처**:
```
┌─────────────────────────────────────────────────────────┐
│                  Citation-Check-Skill                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Pass 1: Response Generation                            │
│  ├─ RAG 시스템 → 초기 응답 + 인용                        │
│                                                         │
│  Pass 2: Citation Verification                          │
│  ├─ 각 인용 추출                                        │
│  ├─ 출처 문서와 대조                                    │
│  ├─ 7-class 분류                                       │
│  └─ 검증 리포트 생성                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**장점**: 세분화된 분류 (7가지), MIT 라이선스, 경량
**단점**: 자동 정정 없음, Stars 적음

---

### 3. Ragas (RAG Assessment)

**GitHub**: [explodinggradients/ragas](https://github.com/explodinggradients/ragas) (8k+ stars)

**핵심 기능**:
- **Faithfulness 메트릭**: 응답이 컨텍스트에 충실한지 (0.0~1.0)
- **Context Precision/Recall**: 검색 품질 평가
- **Answer Relevancy**: 답변 관련성
- **산업 표준**: LlamaIndex, LangChain 공식 통합

**주요 메트릭**:
| 메트릭 | 설명 | 범위 |
|--------|------|------|
| `faithfulness` | 응답이 컨텍스트에만 기반하는지 | 0.0~1.0 |
| `context_precision` | 관련 청크가 상위 랭킹인지 | 0.0~1.0 |
| `context_recall` | 필요한 정보가 모두 검색됐는지 | 0.0~1.0 |
| `answer_relevancy` | 답변이 질문에 적합한지 | 0.0~1.0 |

**사용 예시**:
```python
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision

result = evaluate(
    dataset,
    metrics=[faithfulness, context_precision]
)
```

**장점**: 산업 표준, 광범위한 메트릭, 프레임워크 통합
**단점**: 배치 평가 (실시간 아님), 자동 정정 없음

---

### 4. TruLens

**GitHub**: [truera/trulens](https://github.com/truera/trulens) (2.5k+ stars)

**핵심 기능**:
- **RAG Triad**: Groundedness, Context Relevance, Answer Relevance
- **실시간 피드백**: 애플리케이션 실행 중 평가
- **대시보드**: 웹 UI로 평가 결과 시각화
- **LLM 기반 평가**: GPT-4 등으로 평가 수행

**RAG Triad**:
```
┌─────────────────────────────────────────────────────────┐
│                      RAG Triad                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Context Relevance                                   │
│     └─ 검색된 컨텍스트가 질문과 관련있는가?              │
│                                                         │
│  2. Groundedness                                        │
│     └─ 응답이 컨텍스트에 근거하는가?                     │
│                                                         │
│  3. Answer Relevance                                    │
│     └─ 응답이 질문에 적합한가?                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**장점**: 실시간 피드백, 웹 대시보드, 상세 분석
**단점**: 설정 복잡, LLM 비용

---

### 5. DeepEval

**GitHub**: [confident-ai/deepeval](https://github.com/confident-ai/deepeval) (4k+ stars)

**핵심 기능**:
- **14+ RAG 메트릭**: 가장 포괄적인 메트릭 세트
- **pytest 통합**: CI/CD 파이프라인 통합
- **G-Eval**: GPT-4 기반 평가
- **Hallucination 탐지**: 전용 환각 메트릭

**주요 메트릭**:
| 카테고리 | 메트릭 |
|---------|--------|
| 충실도 | `FaithfulnessMetric`, `HallucinationMetric` |
| 관련성 | `AnswerRelevancyMetric`, `ContextualRelevancyMetric` |
| 품질 | `SummarizationMetric`, `CoherenceMetric` |
| 보안 | `ToxicityMetric`, `BiasMetric` |

**사용 예시**:
```python
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric

test_cases = [...]
evaluate(test_cases, [FaithfulnessMetric()])
```

**장점**: 가장 많은 메트릭, CI/CD 통합, 환각 전용 메트릭
**단점**: 배치 평가, 자동 정정 없음

---

### 6. Arize Phoenix

**GitHub**: [Arize-ai/phoenix](https://github.com/Arize-ai/phoenix) (3k+ stars)

**핵심 기능**:
- **RAG 추적**: 전체 파이프라인 추적
- **프로덕션 모니터링**: 실시간 성능 모니터링
- **평가 통합**: Ragas, TruLens 메트릭 지원
- **시각화**: 대화형 대시보드

**아키텍처**:
```
LLM App → Phoenix Tracing → Evaluation → Dashboard
                    ↓
          Retrieval Analysis
          Context Quality
          Response Quality
```

**장점**: 프로덕션 모니터링, 시각화, 메트릭 통합
**단점**: 인프라 필요, 복잡한 설정

---

### 7. LLMWare

**GitHub**: [llmware-ai/llmware](https://github.com/llmware-ai/llmware) (7k+ stars)

**핵심 기능**:
- **evidence_check_numbers**: 숫자/통계 검증
- **evidence_check_sources**: 출처 정확성 검증
- **경량 모델**: 로컬 실행 가능한 작은 모델
- **RAG 통합**: 자체 RAG 파이프라인

**검증 함수**:
```python
# 숫자 검증
result = model.evidence_check_numbers(
    context="GDP는 2023년 5% 성장...",
    claim="GDP가 5% 성장했다"
)

# 출처 검증
result = model.evidence_check_sources(
    context=retrieved_docs,
    claim=response_claim
)
```

**장점**: 로컬 실행, 경량, 특화 함수
**단점**: 기능 제한적, 커뮤니티 작음

---

### 8. FACTUM (Stanford 연구)

**논문**: [arXiv:2601.05866](https://arxiv.org/abs/2601.05866)

**핵심 기능**:
- **기계적 탐지**: Attention + FFN 경로 분석
- **Citation Hallucination 특화**: 인용 환각 전용
- **학술적 엄밀성**: Stanford NLP 연구

**방법론**:
```
입력 → Attention 경로 추출 → FFN 경로 분석 → 환각 판정
           ↓                      ↓
      인용 의존성 확인        지식 출처 분석
```

**성능**: AUC +37.5% 향상 (기존 방법 대비)

**장점**: 학술적 검증, 기계적 접근
**단점**: 연구용, 프로덕션 미지원

---

### 9. SemanticCite

**특징**:
- **4-class 분류**: Supported, Contradicted, Unverifiable, Partial
- **의미론적 검증**: 문장 임베딩 기반 유사도
- **NLI 통합**: Natural Language Inference 활용

**분류 체계**:
| 클래스 | 설명 |
|--------|------|
| `Supported` | 출처가 주장을 지지 |
| `Contradicted` | 출처가 주장과 모순 |
| `Unverifiable` | 출처에서 확인 불가 |
| `Partial` | 부분적으로 지지 |

---

### 10. HaluGate

**특징**:
- **Token-level 환각 탐지**: 토큰 단위 정밀 탐지
- **빠른 처리**: ~125ms for 4K context
- **실시간 가능**: 프로덕션 사용 가능

**아키텍처**:
```
Response → Token Decomposition → Per-Token Analysis → Hallucination Map
                                         ↓
                              Confidence Score per Token
```

---

## 기능 비교 테이블

### 핵심 기능 비교

| 도구 | 실시간 검증 | 자동 정정 | 상태 분류 | 신뢰도 점수 | 대시보드 |
|-----|-----------|----------|----------|-----------|---------|
| **OKT-RAG** | ✅ | ✅ | 3가지 | ✅ | ✅ |
| **Citation-Check-Skill** | ✅ | ❌ | 7가지 | ✅ | ❌ |
| **Ragas** | ❌ (배치) | ❌ | - | ✅ | ❌ |
| **TruLens** | ✅ | ❌ | - | ✅ | ✅ |
| **DeepEval** | ❌ (배치) | ❌ | - | ✅ | ✅ |
| **Arize Phoenix** | ✅ | ❌ | - | ✅ | ✅ |
| **LLMWare** | ✅ | ❌ | 2가지 | ✅ | ❌ |
| **FACTUM** | ❌ | ❌ | 2가지 | ✅ | ❌ |
| **SemanticCite** | ✅ | ❌ | 4가지 | ✅ | ❌ |
| **HaluGate** | ✅ | ❌ | Token-level | ✅ | ❌ |

### 검증 방식 비교

| 도구 | 검증 방식 | 검증 수준 | 비용 |
|-----|----------|----------|------|
| **OKT-RAG** | LLM 기반 | 문장 레벨 | 중간 (LLM 호출) |
| **Citation-Check-Skill** | LLM 기반 | 인용 레벨 | 중간 |
| **Ragas** | LLM 기반 | 응답 레벨 | 중간 |
| **TruLens** | LLM 기반 | 응답 레벨 | 중간~높음 |
| **DeepEval** | LLM 기반 | 응답 레벨 | 중간 |
| **LLMWare** | 로컬 모델 | 주장 레벨 | 낮음 |
| **FACTUM** | Attention 분석 | 토큰 레벨 | 낮음 |
| **HaluGate** | 모델 분석 | 토큰 레벨 | 낮음 |

### 통합 용이성

| 도구 | 설치 복잡도 | 프레임워크 통합 | 프로덕션 준비 |
|-----|-----------|--------------|-------------|
| **OKT-RAG** | 낮음 | FastAPI | ✅ |
| **Citation-Check-Skill** | 낮음 | 범용 | ✅ |
| **Ragas** | 낮음 | LlamaIndex, LangChain | ✅ |
| **TruLens** | 중간 | LlamaIndex, LangChain | ✅ |
| **DeepEval** | 낮음 | pytest | ✅ |
| **Arize Phoenix** | 높음 | 범용 | ✅ |
| **LLMWare** | 낮음 | 자체 | ✅ |
| **FACTUM** | 높음 | 연구용 | ❌ |

---

## 사용 시나리오별 권장

| 시나리오 | 권장 도구 | 이유 |
|---------|----------|------|
| **실시간 인용 검증 + 자동 정정** | OKT-RAG | 유일한 자동 정정 지원 |
| **세분화된 인용 상태 분류** | Citation-Check-Skill | 7가지 상태 분류 |
| **RAG 파이프라인 품질 측정** | Ragas | 산업 표준, 프레임워크 통합 |
| **개발 중 실시간 피드백** | TruLens | 대시보드, 실시간 피드백 |
| **CI/CD 통합 테스트** | DeepEval | pytest 통합, 14+ 메트릭 |
| **프로덕션 모니터링** | Arize Phoenix | 추적, 모니터링 |
| **로컬 경량 검증** | LLMWare | 로컬 실행, 저비용 |
| **학술 연구** | FACTUM | 기계적 분석, 학술적 엄밀성 |

---

## OKT-RAG vs Citation-Check-Skill 상세 비교

| 항목 | OKT-RAG | Citation-Check-Skill |
|-----|---------|---------------------|
| **검증 방식** | LLM 기반 실시간 | LLM 기반 Two-pass |
| **상태 분류** | 3가지 (Accurate, Inaccurate, Uncertain) | 7가지 (Verified~Not in Source) |
| **자동 정정** | ✅ 지원 | ❌ 미지원 |
| **번호 재정렬** | ✅ 자동 | ❌ 수동 |
| **웹 UI** | ✅ 대시보드 | ❌ |
| **RAG 통합** | ✅ LightRAG 내장 | 외부 RAG 필요 |
| **Stars** | Private | 30+ |
| **라이선스** | MIT | MIT |

**핵심 차이**:
- OKT-RAG: **자동 정정**에 초점, 프로덕션 즉시 사용
- Citation-Check-Skill: **세분화된 분류**에 초점, 분석용

---

## 아키텍처 비교

### 실시간 검증 vs 배치 평가

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    검증 방식 스펙트럼                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [배치 평가]                        [실시간 검증]                         │
│                                                                         │
│  • Ragas                           • OKT-RAG                            │
│  • DeepEval                        • Citation-Check-Skill               │
│  • 사후 품질 측정                    • TruLens                            │
│  • 개발/테스트 단계                  • HaluGate                           │
│  • 수동 개선 필요                    • 프로덕션 사용                        │
│                                    • 자동 정정 가능 (OKT-RAG)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 검증 수준 비교

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    검증 수준 스펙트럼                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [응답 레벨]         [문장 레벨]         [토큰 레벨]                       │
│                                                                         │
│  • Ragas            • OKT-RAG           • FACTUM                        │
│  • TruLens          • Citation-Check    • HaluGate                      │
│  • DeepEval                                                             │
│                                                                         │
│  정밀도: 낮음 ←────────────────────────────────→ 높음                     │
│  속도:   빠름 ←────────────────────────────────→ 느림                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 결론

### 기능별 1위 도구

| 카테고리 | 1위 | 이유 |
|---------|-----|------|
| **실시간 검증 + 자동 정정** | OKT-RAG | 유일한 자동 정정 지원 |
| **상태 분류 세분화** | Citation-Check-Skill | 7가지 상태 |
| **산업 표준 메트릭** | Ragas | 가장 널리 사용 |
| **CI/CD 통합** | DeepEval | pytest 네이티브 |
| **프로덕션 모니터링** | Arize Phoenix | 추적 + 대시보드 |
| **로컬 경량 검증** | LLMWare | 로컬 모델 지원 |
| **학술적 엄밀성** | FACTUM | 기계적 분석 |

### OKT-RAG가 적합한 경우

- 실시간 인용 검증 + 자동 정정이 필요한 경우
- 프로덕션에서 즉시 사용 가능한 솔루션이 필요한 경우
- 팩트체크가 중요한 도메인 (학술, 법률, 의료)
- 웹 대시보드로 검증 결과를 시각화해야 하는 경우

### 다른 도구가 적합한 경우

- 세분화된 인용 상태 분류 → Citation-Check-Skill
- RAG 파이프라인 품질 측정 → Ragas
- CI/CD 자동화 테스트 → DeepEval
- 프로덕션 모니터링 → Arize Phoenix
- 학술 연구 → FACTUM

---

## 참고 자료

### GitHub 저장소

- [Citation-Check-Skill](https://github.com/serenakeyitan/Citation-Check-Skill)
- [Ragas](https://github.com/explodinggradients/ragas)
- [TruLens](https://github.com/truera/trulens)
- [DeepEval](https://github.com/confident-ai/deepeval)
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)
- [LLMWare](https://github.com/llmware-ai/llmware)

### 학술 논문

- [FACTUM: Mechanistic Detection of Citation Hallucination](https://arxiv.org/abs/2601.05866)
- [HalluGraph: Auditable Hallucination Detection](https://arxiv.org/abs/2512.01659)
- [MedRAGChecker: Claim-Level Verification](https://arxiv.org/html/2601.06519)
- [RAC: Retrieval Augmented Correction](https://arxiv.org/html/2410.15667)

### 문서

- [Ragas Faithfulness Metric](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [TruLens RAG Triad](https://www.trulens.org/trulens/getting_started/quickstarts/rag_triad/)
- [DeepEval RAG Metrics](https://docs.confident-ai.com/docs/metrics-rag)
