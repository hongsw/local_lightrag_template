# OKT-RAG (Open Korea Tech RAG)

> **"모델에 종속되지 않고, 실사용 로그가 곧 연구 데이터가 되는 한국형 차세대 RAG 플랫폼"**

LightRAG 기반 Knowledge Graph + Vector 검색을 지원하는 RAG(Retrieval-Augmented Generation) API 서버

**Version: 0.4.1** | [변경 이력](#변경-이력)

![OKT-RAG Dashboard](docs/images/image1.png)

## 주요 기능

### 핵심 기능
- **Knowledge Graph RAG**: LightRAG 엔진으로 벡터 검색과 Knowledge Graph를 결합
- **다중 쿼리 모드**: naive, local, global, hybrid, **adaptive** 모드 지원
- **🆕 자동 인용 검증**: AI 기반 인용 정확도 검증 및 자동 정정
- **🆕 인라인 인용 스타일링**: 클릭 가능한 인용 배지 `[†1]` → 소스로 이동

### OKT-RAG v0.4.0 신규 기능
- **🆕 Multi-Embedding**: 문서 1개를 여러 임베딩 공간에 동시 저장 (semantic 1536D, fast 512D)
- **🆕 Matryoshka 지원**: 품질/속도 트레이드오프를 위한 차원 축소
- **🆕 Retrieval Observability**: 모든 검색 의사결정 과정 완전 로그화
- **🆕 Adaptive Retrieval**: 질문 유형 자동 분류 및 최적 전략 선택
- **🆕 Analytics Dashboard**: 슬롯별/전략별 성능 분석, 비용 추적

### 인프라
- **다양한 데이터 소스**: Local Files, Google Drive, Slack, Notion 커넥터
- **증분 인덱싱**: 새 파일만 자동 감지하여 인덱싱 (중복 방지)
- **FastAPI 기반**: 자동 API 문서화 (Swagger UI)
- **컨테이너 지원**: Docker, Docker Compose, Kubernetes 배포 지원

## 요구 사항

- Python 3.12+
- OpenAI API Key

## 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone <repository-url>
cd korea_tech_edu_rag_pdf

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 설정

`.env` 파일을 프로젝트 루트에 생성:

```bash
# 필수
OPENAI_API_KEY=sk-your-api-key

# 선택: 모델 설정
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# 선택: 경로 설정
LOCAL_FILES_PATH=./rag_raw_pdfs
LIGHTRAG_WORKING_DIR=./lightrag_data
```

### 3. PDF 파일 준비

```bash
mkdir rag_raw_pdfs
# PDF 파일을 rag_raw_pdfs 폴더에 복사
```

### 4. 서버 실행

```bash
python run.py
```

### 5. 대시보드 접속

브라우저에서 `http://localhost:8000/dashboard` 접속

## 웹 대시보드 사용법

### 기본 쿼리

1. Query Mode 선택 (Hybrid 또는 **Adaptive** 권장)
2. Embedding Slot 선택 (선택사항)
3. 질문 입력
4. Send 버튼 클릭

### Auto-Verify Mode (자동 검증)

**Auto-Verify Mode** 토글이 활성화되면:
1. RAG 응답 생성
2. 각 인용 [†N]을 소스와 자동 대조
3. 부정확한 인용 자동 제거
4. 정정된 응답 + 검증 로그 표시

![인라인 인용과 소스 검증](docs/images/image2.png)

### 인용 검증 결과

검증 후 각 인용에 대해 정확/부정확/불확실 상태가 표시됩니다.

![인용 검증 결과와 인덱싱](docs/images/image3.png)

### 인덱싱

- **Sync**: 새로운/변경된 파일만 인덱싱
- **Rebuild**: 전체 인덱스 재구축

## API 사용법

### 기본 쿼리

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "데이터 엔지니어링이란?", "mode": "hybrid"}'
```

### 검증된 쿼리 (자동 인용 검증)

```bash
curl -X POST "http://localhost:8000/query/verified" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "미국 제조업 부흥 정책의 제약요인은?",
    "mode": "hybrid",
    "confidence_threshold": 0.7
  }'
```

**응답 예시**:
```json
{
  "original_answer": "원본 응답 (검증 전)",
  "corrected_answer": "정정된 응답 (부정확한 인용 제거됨)",
  "accuracy_rate": 0.8,
  "removed_citations": [3, 5],
  "verification_log": [
    {
      "citation_number": 1,
      "statement": "인용된 문장",
      "is_accurate": true,
      "confidence": 0.9,
      "explanation": "출처 내용과 일치합니다"
    }
  ]
}
```

### 수동 인용 검증

```bash
curl -X POST "http://localhost:8000/query/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "answer": "검증할 응답 텍스트[†1]",
    "sources": [{"file_name": "doc.pdf", "page": 10, "excerpt": "소스 내용"}]
  }'
```

## 인용 형식

본 시스템은 `[†N]` 형식의 인용을 사용합니다:

```markdown
미국 제조업은 구조적 위기를 겪고 있습니다[†1].
높은 인건비로 인해 자국 생산의 매력도가 저하되고 있습니다[†2].

### References
- [†1] 제조업 혁신 방안.pdf, p.100
- [†2] 미국 제조업 정책.pdf, p.2
```

## API 엔드포인트

### Query
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/query` | RAG 쿼리 (Adaptive 모드 지원) |
| POST | `/query/verified` | 검증된 RAG 쿼리 (자동 인용 검증) |
| POST | `/query/verify` | 수동 인용 검증 |
| GET | `/query/modes` | 쿼리 모드 목록 |

### Multi-Embedding (🆕 v0.4.0)
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/embedding/config` | 멀티 임베딩 설정 조회 |
| GET | `/embedding/slots` | 임베딩 슬롯 목록 |
| GET | `/embedding/matryoshka` | Matryoshka 차원 정보 |

### Analytics (🆕 v0.4.0)
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/analytics/retrieval` | 검색 통계 (기간별) |
| GET | `/analytics/slots/performance` | 슬롯별 성능 분석 |
| GET | `/analytics/strategies/performance` | 전략별 성능 분석 |
| GET | `/analytics/cost` | 비용 분석 |
| GET | `/analytics/accuracy/trend` | 정확도 추이 |

### Index
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/index/local` | 로컬 파일 인덱싱 |
| POST | `/index/local/sync` | 새 파일만 인덱싱 (증분) |
| POST | `/index/rebuild` | 전체 인덱스 재구축 |
| GET | `/index/files` | 인덱싱된 파일 목록 |

### System
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/stats` | 시스템 통계 |
| GET | `/health` | 헬스 체크 |
| GET | `/dashboard` | 웹 대시보드 |

자세한 API 사용법은 [API_USAGE.md](docs/API_USAGE.md) 참조

## 쿼리 모드

| 모드 | 설명 | 용도 |
|------|------|------|
| `naive` | 단순 벡터 유사도 검색 | 빠른 검색, 간단한 질문 |
| `local` | Knowledge Graph 로컬 엔티티 관계 | 특정 개념 간 관계 파악 |
| `global` | Knowledge Graph 글로벌 패턴 | 전체적인 맥락 이해 |
| `hybrid` | local + global 결합 | 대부분의 질문 |
| `adaptive` | 🆕 질문 유형 자동 분류 + 최적 전략 선택 (권장) | 모든 질문 유형 |

### Adaptive 모드 (v0.4.0)

질문을 자동 분류하여 최적의 검색 전략을 선택합니다:

| 질문 유형 | 검색 전략 | 임베딩 슬롯 |
|----------|----------|------------|
| Factual (사실 확인) | Dense | semantic |
| Analytical (분석) | Hybrid | semantic + keyword |
| Comparative (비교) | Multi-slot | all slots |

## 프로젝트 구조

```
korea_tech_edu_rag_pdf/
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI 앱 진입점
│   │   ├── models.py         # Pydantic 모델
│   │   └── routes/
│   │       ├── query.py      # 쿼리 + 검증 엔드포인트
│   │       ├── index.py      # 인덱싱 엔드포인트
│   │       └── sources.py    # 소스 관리 엔드포인트
│   ├── connectors/           # 데이터 소스 커넥터
│   ├── config.py             # 환경 설정
│   ├── index_tracker.py      # 인덱스 추적
│   └── rag_engine.py         # LightRAG 래퍼 + 검증 로직
├── docs/
│   ├── API_USAGE.md          # API 상세 문서
│   ├── DEPLOYMENT.md         # 배포 가이드
│   ├── RAG_CITATION_PROCESS.md  # 🆕 인용 검증 프로세스 문서
│   └── PROMPTS.md            # 🆕 프롬프트 문서
├── dashboard.html            # 🆕 웹 대시보드
├── k8s/                      # Kubernetes 매니페스트
├── run.py                    # 실행 스크립트
└── requirements.txt          # 의존성
```

## Docker로 실행

```bash
# .env 파일 생성 (OPENAI_API_KEY 필수)
echo "OPENAI_API_KEY=sk-your-api-key" > .env

# Docker Compose로 실행
docker compose up -d

# 로그 확인
docker compose logs -f
```

## 타 프레임워크와 비교 (10개)

> GitHub Stars 실시간 조회 기준 (2025-01-27)

| 프레임워크 | GitHub Stars | Knowledge Graph | 인용 검증 | 자동 정정 | 웹 UI |
|-----------|-------------|-----------------|----------|----------|-------|
| **Korea Tech RAG** | Private | ✅ (경량) | ✅ | ✅ | ✅ 한글 |
| Dify | 127k+ | ❌ | ❌ | ❌ | ✅ |
| LangChain | 125k+ | 플러그인 | ❌ | ❌ | ❌ |
| RAGFlow | 72k+ | ❌ | ❌ | ❌ | ✅ |
| LlamaIndex | 47k+ | 플러그인 | ❌ | ❌ | ❌ |
| DSPy | 32k+ | ❌ | ❌ | ❌ | ❌ |
| GraphRAG | 31k+ | ✅ (전체) | ❌ | ❌ | ❌ |
| LightRAG | 28k+ | ✅ (경량) | ❌ | ❌ | ❌ |
| Kotaemon | 25k+ | GraphRAG 통합 | ❌ | ❌ | ✅ |
| Haystack | 24k+ | ❌ | ❌ | ❌ | 제한적 |

### 기능별 1위 프레임워크

| 카테고리 | 1위 | 이유 |
|---------|-----|------|
| **인용 정확성** | Korea Tech RAG | 유일한 자동 인용 검증+정정 |
| **Knowledge Graph** | GraphRAG | Leiden Community Detection |
| **사용 편의성** | Dify | 로우코드, 풀 UI, 127k+ stars |
| **생태계** | LangChain | 125k+ stars |
| **문서 파싱** | RAGFlow | 심층 문서 이해 (DeepDoc) |

### 차별점

1. **자동 인용 검증**: 응답의 각 인용을 소스와 대조하여 정확성 검증
2. **자동 정정**: 부정확한 인용을 자동으로 제거하고 번호 재정렬
3. **엄격한 소스 기반**: 프롬프트 최적화로 환각(hallucination) 최소화
4. **통합 대시보드**: 쿼리, 검증, 인덱싱을 한 곳에서 관리

### 적합한 사용 사례

- 📚 학술/연구 문서 기반 QA
- 📋 정확한 출처 표기가 필요한 리포트 생성
- ✅ 팩트체크가 중요한 도메인
- 🎓 교육 자료 기반 질의응답

자세한 10개 프레임워크 비교는 [COMPARISON.md](docs/COMPARISON.md) 참조

## 기술 스택

- **RAG 엔진**: [LightRAG](https://github.com/HKUDS/LightRAG) - Knowledge Graph + Vector
- **문서 로더**: [LlamaIndex](https://www.llamaindex.ai/) - 다양한 데이터 소스 지원
- **API 프레임워크**: [FastAPI](https://fastapi.tiangolo.com/)
- **LLM**: OpenAI GPT-4o-mini (기본)
- **임베딩**: OpenAI text-embedding-3-small (기본)
- **대시보드**: Tailwind CSS + Vanilla JS

## 문서

- [API 사용법](docs/API_USAGE.md)
- [배포 가이드](docs/DEPLOYMENT.md)
- [인용 검증 프로세스](docs/RAG_CITATION_PROCESS.md)
- [프롬프트 문서](docs/PROMPTS.md)
- [타 프레임워크 비교](docs/COMPARISON.md)

## 변경 이력

### v0.4.1 (2025-01-28)
#### Analytics Dashboard 확장
- 🆕 Slot Performance 차트: 슬롯별 응답시간/쿼리수 비교
- 🆕 Strategy Performance 차트: 전략별 응답시간/정확도 비교
- 🆕 Accuracy Trend 차트: 인용 정확도 추이 라인 차트
- 🆕 Query Type Distribution 차트: Factual/Analytical/Comparative 분포

#### 버그 수정
- 🔧 검증 후 제거된 인용의 소스가 Sources 영역에서 필터링되도록 수정
- 🔧 References 섹션과 Sources 섹션 개수 불일치 문제 해결

### v0.4.0 (2025-01-27) - OKT-RAG 릴리스
**"Open Korea Tech RAG - Model-Agnostic, Multi-Embedding, Observable"**

#### Multi-Embedding Native
- 🆕 Multi-Embedding Store: 문서 1개를 여러 임베딩 공간에 동시 저장
- 🆕 Matryoshka 지원: OpenAI text-embedding-3-small 512D/1536D 차원 선택
- 🆕 임베딩 슬롯 API: `/embedding/config`, `/embedding/slots`

#### Retrieval Observability
- 🆕 완전 로그화: 모든 검색 의사결정 과정 기록
- 🆕 RetrievalLogger: 쿼리별 임베딩/검색/응답 단계 로깅
- 🆕 Analytics API: `/analytics/retrieval`, `/analytics/cost`
- 🆕 Analytics Dashboard: 슬롯별, 전략별 성능 차트 시각화

#### Adaptive Retrieval
- 🆕 질문 유형 분류: Factual, Analytical, Comparative 자동 분류
- 🆕 전략 자동 선택: 질문 유형에 따른 최적 검색 전략
- 🆕 Adaptive 쿼리 모드 추가

#### 대시보드 개선
- 🆕 탭 네비게이션: Query / Multi-Embedding / Analytics / Files
- 🆕 인라인 인용 스타일링: 클릭 가능한 보라색 배지
- 🆕 Quick Stats 사이드바: 실시간 성능 메트릭
- 🆕 OKT-RAG Features 표시: Observability, Multi-Embedding, Adaptive 상태

### v0.3.0 (2025-01-27)
- 🆕 자동 인용 검증 기능 (`/query/verified` 엔드포인트)
- 🆕 수동 인용 검증 기능 (`/query/verify` 엔드포인트)
- 🆕 웹 대시보드 (Auto-Verify Mode 포함)
- 🆕 인용 형식 변경: `[1]` → `[†1]`
- 🆕 소스 관련성 점수 기반 필터링 (최대 5개)
- 📝 프로세스 문서화 (`RAG_CITATION_PROCESS.md`, `PROMPTS.md`)

### v0.2.0
- 소스 인용 및 Markdown 미리보기 추가
- References 섹션 자동 생성

### v0.1.0
- 초기 릴리스
- LightRAG 기반 RAG 엔진
- FastAPI REST API
- Docker/Kubernetes 지원

## 라이선스

MIT License
