# OKT-RAG (Open Korea Tech RAG)

모델에 종속되지 않고, 실사용 로그가 곧 연구 데이터가 되는 한국형 차세대 RAG 플랫폼.

## 핵심 설계 철학

| 철학 | 설명 |
|-----|------|
| Model-Agnostic | 특정 LLM/임베딩에 묶이지 않음 |
| Multi-Embedding Native | 문서 1개를 여러 임베딩 공간에 동시 저장 |
| Retrieval is Observable | 검색 의사결정 과정을 완전 로그화 |
| Adaptive Retrieval | 질문 유형에 따른 동적 전략 선택 |

## 개발 명령어

```bash
# 서버 실행
python run.py serve

# 테스트 실행
pytest tests/

# 린트
ruff check src/

# 의존성 설치
pip install -r requirements.txt
```

## 프로젝트 구조

```
src/
├── api/
│   ├── main.py              # FastAPI 앱 진입점
│   └── routes/
│       ├── query.py         # RAG 쿼리 엔드포인트
│       ├── index.py         # 인덱싱 엔드포인트
│       ├── embedding.py     # 임베딩 설정 API
│       └── analytics.py     # 분석 API
├── embeddings/
│   ├── multi_store.py       # 멀티 임베딩 저장소
│   └── providers/
│       ├── base.py          # 임베딩 인터페이스
│       └── openai.py        # OpenAI (Matryoshka 지원)
├── llm/
│   └── providers/
│       ├── base.py          # LLM 인터페이스
│       └── openai.py        # OpenAI GPT
├── observability/
│   ├── retrieval_logger.py  # 검색 로거
│   └── analytics.py         # 분석 모듈
├── retrieval/
│   └── adaptive.py          # 적응형 검색
├── connectors/              # 데이터 소스 커넥터
├── config.py                # 설정 (멀티 임베딩 슬롯 포함)
├── rag_engine.py            # LightRAG 래퍼
└── processor.py             # 문서 청킹
```

## 주요 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | /query | RAG 쿼리 실행 |
| POST | /query/verified | 인용 검증 포함 쿼리 |
| POST | /index/local | 로컬 파일 인덱싱 |
| GET | /embedding/config | 멀티 임베딩 설정 조회 |
| GET | /embedding/slots | 임베딩 슬롯 목록 |
| GET | /analytics/retrieval | 검색 분석 데이터 |
| GET | /analytics/cost | 비용 분석 |

## 멀티 임베딩 슬롯

기본 설정 (config.py):
- `semantic`: 1536D (OpenAI text-embedding-3-small 풀 버전)
- `semantic_fast`: 512D (Matryoshka 축소 버전, 빠른 검색용)

## 환경 설정

`.env` 파일:
```
OPENAI_API_KEY=sk-...
MULTI_EMBEDDING_ENABLED=true
DEFAULT_EMBEDDING_SLOT=semantic
```

Python 3.12+ 권장
