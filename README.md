# Korea Tech Education RAG

LightRAG 기반 Knowledge Graph + Vector 검색을 지원하는 RAG(Retrieval-Augmented Generation) API 서버

## 주요 기능

- **Knowledge Graph RAG**: LightRAG 엔진으로 벡터 검색과 Knowledge Graph를 결합
- **다중 쿼리 모드**: naive, local, global, hybrid 모드 지원
- **다양한 데이터 소스**: Local Files, Google Drive, Slack, Notion 커넥터
- **FastAPI 기반**: 자동 API 문서화 (Swagger UI)
- **문서 청킹**: 효율적인 문서 분할 및 처리
- **컨테이너 지원**: Docker, Docker Compose, Kubernetes 배포 지원

## 요구 사항

- Python 3.12+
- OpenAI API Key

## 설치

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

## 환경 설정

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

# 선택: 청킹 설정
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# 선택: Google Drive
GOOGLE_CREDENTIALS_PATH=./credentials.json
GOOGLE_DRIVE_FOLDER_ID=1ABC...

# 선택: Slack
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNEL_IDS=C01234567,C89012345

# 선택: Notion
NOTION_API_KEY=secret_...
NOTION_DATABASE_IDS=abc123,def456
```

## 빠른 시작

### 1. PDF 파일 준비

```bash
mkdir rag_raw_pdfs
# PDF 파일을 rag_raw_pdfs 폴더에 복사
```

### 2. 서버 실행

```bash
# 기본 실행
python run.py

# 파일 인덱싱과 함께 실행
python run.py --index --path ./rag_raw_pdfs
```

### 3. API 사용

```bash
# 로컬 파일 인덱싱
curl -X POST "http://localhost:8000/index/local" \
  -H "Content-Type: application/json" \
  -d '{"path": "./rag_raw_pdfs", "recursive": true}'

# RAG 쿼리
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "데이터 엔지니어링이란?", "mode": "hybrid"}'
```

### 4. API 문서

브라우저에서 `http://localhost:8000/docs` 접속

## Docker로 실행

```bash
# .env 파일 생성 (OPENAI_API_KEY 필수)
echo "OPENAI_API_KEY=sk-your-api-key" > .env

# Docker Compose로 실행
docker compose up -d

# 로그 확인
docker compose logs -f

# 중지
docker compose down
```

### 수동 Docker 실행

```bash
# 이미지 빌드
docker build -t lightrag-api:latest .

# 컨테이너 실행
docker run -d \
  --name lightrag-api \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-api-key \
  -v $(pwd)/rag_raw_pdfs:/app/rag_raw_pdfs:ro \
  -v lightrag_data:/app/lightrag_data \
  lightrag-api:latest
```

## Kubernetes 배포

```bash
# Secret 설정 (k8s/secret.yaml 수정 또는)
kubectl create namespace lightrag
kubectl -n lightrag create secret generic lightrag-secret \
  --from-literal=OPENAI_API_KEY=sk-your-api-key

# Kustomize로 배포
kubectl apply -k k8s/

# 상태 확인
kubectl -n lightrag get all

# 포트 포워딩
kubectl -n lightrag port-forward svc/lightrag-api 8000:80
```

자세한 배포 가이드는 [DEPLOYMENT.md](docs/DEPLOYMENT.md) 참조

## 프로젝트 구조

```
korea_tech_edu_rag_pdf/
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI 앱 진입점
│   │   ├── models.py         # Pydantic 모델
│   │   └── routes/
│   │       ├── query.py      # 쿼리 엔드포인트
│   │       ├── index.py      # 인덱싱 엔드포인트
│   │       └── sources.py    # 소스 관리 엔드포인트
│   ├── connectors/
│   │   ├── base.py           # 커넥터 베이스 클래스
│   │   ├── local_files.py    # 로컬 파일 커넥터
│   │   ├── google_drive.py   # Google Drive 커넥터
│   │   ├── slack.py          # Slack 커넥터
│   │   └── notion.py         # Notion 커넥터
│   ├── config.py             # 환경 설정
│   ├── processor.py          # 문서 청킹
│   └── rag_engine.py         # LightRAG 래퍼
├── k8s/                      # Kubernetes 매니페스트
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── pvc.yaml
│   ├── ingress.yaml
│   └── kustomization.yaml
├── docs/
│   ├── API_USAGE.md          # API 상세 문서
│   └── DEPLOYMENT.md         # 배포 가이드
├── rag_raw_pdfs/             # PDF 파일 디렉토리
├── lightrag_data/            # LightRAG 데이터 저장소
├── Dockerfile                # Docker 이미지 빌드
├── docker-compose.yml        # Docker Compose 설정
├── Makefile                  # 편의 명령어
├── run.py                    # 실행 스크립트
├── requirements.txt          # 의존성
└── .env                      # 환경 변수 (생성 필요)
```

## 쿼리 모드

| 모드 | 설명 | 용도 |
|------|------|------|
| `naive` | 단순 벡터 유사도 검색 | 빠른 검색, 간단한 질문 |
| `local` | Knowledge Graph 로컬 엔티티 관계 | 특정 개념 간 관계 파악 |
| `global` | Knowledge Graph 글로벌 패턴 | 전체적인 맥락 이해 |
| `hybrid` | local + global 결합 (권장) | 대부분의 질문 |

## LightRAG 동작 원리

LightRAG는 기존 RAG와 달리 **Knowledge Graph**를 활용하여 문서 간 관계를 파악하고 더 정확한 답변을 생성합니다.

### 1. 인덱싱 단계

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐
│   PDF 문서   │ → │  텍스트 추출  │ → │   LLM 엔티티 추출    │
└─────────────┘    └──────────────┘    └─────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Graph 생성                      │
│  • 엔티티(Entity): 개념, 기술, 방법론 등                      │
│  • 관계(Relation): 엔티티 간의 연결 관계                      │
│  • 청크(Chunk): 원본 텍스트 조각 + 벡터 임베딩                │
└─────────────────────────────────────────────────────────────┘
```

### 2. 데이터 구조

| 파일 | 설명 | 내용 |
|------|------|------|
| `graph_chunk_entity_relation.graphml` | Knowledge Graph | 노드(엔티티) + 엣지(관계) |
| `vdb_entities.json` | 엔티티 벡터 DB | 엔티티 임베딩 + 메타데이터 |
| `vdb_relationships.json` | 관계 벡터 DB | 관계 임베딩 + 메타데이터 |
| `vdb_chunks.json` | 청크 벡터 DB | 텍스트 청크 임베딩 |
| `kv_store_*.json` | Key-Value 저장소 | 원본 텍스트, 설명 등 |

### 3. 쿼리 단계 (Dual-Level Retrieval)

```
┌──────────────────────────────────────────────────────────────┐
│                        사용자 질문                            │
│            "LightRAG의 dual-level retrieval이란?"            │
└──────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   Low-Level Retrieval   │     │  High-Level Retrieval   │
│   (local 모드)          │     │   (global 모드)         │
├─────────────────────────┤     ├─────────────────────────┤
│ • 특정 엔티티 검색       │     │ • 커뮤니티/테마 검색     │
│ • 직접 관계 탐색         │     │ • 글로벌 패턴 분석       │
│ • 연결된 청크 수집       │     │ • 상위 개념 파악         │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    컨텍스트 통합 (hybrid)                     │
│  Low-Level 정보 + High-Level 정보 → 종합적인 컨텍스트         │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                     LLM 답변 생성                             │
│  통합된 컨텍스트를 바탕으로 정확하고 포괄적인 답변 생성        │
└──────────────────────────────────────────────────────────────┘
```

### 4. 기존 RAG vs LightRAG

| 구분 | 기존 RAG | LightRAG |
|------|----------|----------|
| 검색 방식 | 벡터 유사도만 사용 | 벡터 + Knowledge Graph |
| 관계 파악 | 불가능 | 엔티티 간 관계 추론 가능 |
| 컨텍스트 | 독립된 청크들 | 연결된 지식 그래프 |
| 답변 품질 | 단편적 | 맥락을 고려한 종합적 답변 |
| 적합한 질문 | 단순 검색 | 개념 간 관계, 비교, 분석 |

### 5. 실제 쿼리 예시

```bash
# 질문: "LightRAG의 graph-based indexing이란?"

# 1단계: 관련 엔티티 검색
#   → "LightRAG", "graph-based indexing", "knowledge graph" 등

# 2단계: 관계 탐색
#   → LightRAG --uses--> graph-based indexing
#   → graph-based indexing --enables--> entity extraction

# 3단계: 컨텍스트 수집
#   → 42개 엔티티, 40개 관계, 20개 청크 검색

# 4단계: LLM이 통합된 컨텍스트로 답변 생성
```

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/query` | RAG 쿼리 |
| POST | `/index/local` | 로컬 파일 인덱싱 |
| POST | `/index/google-drive` | Google Drive 인덱싱 |
| POST | `/index/slack` | Slack 채널 인덱싱 |
| POST | `/index/notion` | Notion 페이지 인덱싱 |
| DELETE | `/index/clear` | 인덱스 초기화 |
| GET | `/sources` | 소스 목록 |
| GET | `/stats` | 시스템 통계 |
| GET | `/health` | 헬스 체크 |

자세한 API 사용법은 [API_USAGE.md](docs/API_USAGE.md) 참조

## 개발

```bash
# 테스트 실행
pytest tests/

# 린트
ruff check src/

# 타입 체크
mypy src/
```

### Makefile 명령어

```bash
make help           # 도움말
make install        # 의존성 설치
make dev            # 개발 서버 실행
make test           # 테스트 실행
make lint           # 린트 실행
make docker-build   # Docker 이미지 빌드
make docker-run     # Docker Compose 실행
make docker-stop    # Docker Compose 중지
make k8s-deploy     # Kubernetes 배포
make k8s-delete     # Kubernetes 삭제
make health         # 헬스체크
```

## 선택적 커넥터 설치

```bash
# Google Drive
pip install llama-index-readers-google

# Slack
pip install llama-index-readers-slack

# Notion
pip install llama-index-readers-notion
```

## 기술 스택

- **RAG 엔진**: [LightRAG](https://github.com/HKUDS/LightRAG) - Knowledge Graph + Vector
- **문서 로더**: [LlamaIndex](https://www.llamaindex.ai/) - 다양한 데이터 소스 지원
- **API 프레임워크**: [FastAPI](https://fastapi.tiangolo.com/)
- **LLM**: OpenAI GPT-4o-mini (기본)
- **임베딩**: OpenAI text-embedding-3-small (기본)

## 라이선스

MIT License
