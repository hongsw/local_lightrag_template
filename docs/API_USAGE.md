# API 사용법

LlamaHub + LightRAG API 서버 사용 가이드

## 시작하기

### 서버 실행

```bash
python run.py serve
```

서버가 시작되면 기본적으로 `http://localhost:8000`에서 실행됩니다.

### API 문서 (Swagger UI)

브라우저에서 `http://localhost:8000/docs` 접속

---

## 엔드포인트 목록

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | API 정보 |
| GET | `/health` | 헬스 체크 |
| GET | `/stats` | 시스템 통계 |
| GET | `/sources` | 사용 가능한 소스 목록 |
| GET | `/sources/local/files` | 로컬 파일 목록 |
| POST | `/query` | RAG 쿼리 |
| GET | `/query/modes` | 쿼리 모드 목록 |
| POST | `/index/local` | 로컬 파일 인덱싱 |
| POST | `/index/google-drive` | Google Drive 인덱싱 |
| POST | `/index/slack` | Slack 채널 인덱싱 |
| POST | `/index/notion` | Notion 페이지 인덱싱 |
| DELETE | `/index/clear` | 인덱스 초기화 |

---

## 쿼리 API

### POST /query

RAG 시스템에 질문을 보내고 답변을 받습니다.

**Request Body:**
```json
{
  "question": "데이터 엔지니어링이란 무엇인가요?",
  "mode": "hybrid"
}
```

**Parameters:**
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| question | string | O | 질문 내용 |
| mode | string | X | 쿼리 모드 (기본값: hybrid) |

**쿼리 모드:**
| 모드 | 설명 |
|------|------|
| naive | 단순 벡터 유사도 검색. 빠르지만 컨텍스트 인식이 낮음 |
| local | Knowledge Graph의 로컬 엔티티 관계 활용 |
| global | 전체 Knowledge Graph의 글로벌 패턴 활용 |
| hybrid | local + global 결합 (권장) |

**cURL 예시:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "RAG란 무엇인가요?",
    "mode": "hybrid"
  }'
```

**Response:**
```json
{
  "answer": "RAG(Retrieval-Augmented Generation)는...",
  "mode": "hybrid",
  "sources": [],
  "metadata": {},
  "processing_time_ms": 1523.45
}
```

### GET /query/modes

사용 가능한 쿼리 모드 목록을 반환합니다.

```bash
curl "http://localhost:8000/query/modes"
```

---

## 인덱싱 API

### POST /index/local

로컬 디렉토리의 파일들(PDF, TXT, MD)을 인덱싱합니다.

**Request Body:**
```json
{
  "path": "./rag_raw_pdfs",
  "recursive": true
}
```

**Parameters:**
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| path | string | O | 파일이 있는 디렉토리 경로 |
| recursive | boolean | X | 하위 디렉토리 포함 여부 (기본값: true) |

**cURL 예시:**
```bash
curl -X POST "http://localhost:8000/index/local" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "./rag_raw_pdfs",
    "recursive": true
  }'
```

**Response:**
```json
{
  "success": true,
  "source_type": "local_files",
  "documents_indexed": 150,
  "documents_failed": 0,
  "error_details": [],
  "message": "Successfully indexed 150 document chunks from 5 files"
}
```

### POST /index/google-drive

Google Drive 폴더의 문서를 인덱싱합니다.

**Request Body:**
```json
{
  "folder_id": "1ABC...xyz",
  "credentials_path": "./credentials.json"
}
```

**Parameters:**
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| folder_id | string | O | Google Drive 폴더 ID |
| credentials_path | string | X | 인증 JSON 파일 경로 (환경변수 사용 시 생략) |

**cURL 예시:**
```bash
curl -X POST "http://localhost:8000/index/google-drive" \
  -H "Content-Type: application/json" \
  -d '{
    "folder_id": "1ABC123xyz"
  }'
```

### POST /index/slack

Slack 채널의 메시지를 인덱싱합니다.

**Request Body:**
```json
{
  "channel_ids": ["C01234567", "C89012345"],
  "token": "xoxb-..."
}
```

**Parameters:**
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| channel_ids | string[] | O | Slack 채널 ID 목록 |
| token | string | X | Slack Bot 토큰 (환경변수 사용 시 생략) |

**cURL 예시:**
```bash
curl -X POST "http://localhost:8000/index/slack" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_ids": ["C01234567"]
  }'
```

### POST /index/notion

Notion 페이지/데이터베이스를 인덱싱합니다.

**Request Body:**
```json
{
  "database_ids": ["abc123..."],
  "page_ids": ["def456..."],
  "token": "secret_..."
}
```

**Parameters:**
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| database_ids | string[] | X | Notion 데이터베이스 ID 목록 |
| page_ids | string[] | X | Notion 페이지 ID 목록 |
| token | string | X | Notion Integration 토큰 (환경변수 사용 시 생략) |

**cURL 예시:**
```bash
curl -X POST "http://localhost:8000/index/notion" \
  -H "Content-Type: application/json" \
  -d '{
    "database_ids": ["abc123def456"]
  }'
```

### DELETE /index/clear

모든 인덱싱된 데이터를 삭제합니다.

```bash
curl -X DELETE "http://localhost:8000/index/clear"
```

**Response:**
```json
{
  "success": true,
  "message": "Index cleared successfully"
}
```

---

## 소스 관리 API

### GET /sources

사용 가능한 데이터 소스 커넥터 목록을 반환합니다.

```bash
curl "http://localhost:8000/sources"
```

**Response:**
```json
{
  "sources": [
    {
      "name": "Local Files",
      "source_type": "local_files",
      "description": "Load documents from local filesystem",
      "is_configured": true,
      "config_requirements": ["LOCAL_FILES_PATH"]
    },
    {
      "name": "Google Drive",
      "source_type": "google_drive",
      "description": "Load documents from Google Drive",
      "is_configured": false,
      "config_requirements": ["GOOGLE_CREDENTIALS_PATH", "GOOGLE_DRIVE_FOLDER_ID"]
    }
  ],
  "total": 4
}
```

### GET /sources/local/files

로컬 파일 디렉토리의 지원되는 파일 목록을 반환합니다.

```bash
curl "http://localhost:8000/sources/local/files"
```

**Response:**
```json
{
  "files": [
    {
      "name": "document.pdf",
      "path": "./rag_raw_pdfs/document.pdf",
      "size": 1024000,
      "extension": ".pdf"
    }
  ],
  "total": 5,
  "base_path": "./rag_raw_pdfs"
}
```

---

## 시스템 API

### GET /health

시스템 상태를 확인합니다.

```bash
curl "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-26T10:30:00.000Z",
  "version": "0.1.0",
  "components": {
    "rag_engine": true,
    "openai_configured": true,
    "local_files_path": true
  }
}
```

### GET /stats

시스템 통계를 반환합니다.

```bash
curl "http://localhost:8000/stats"
```

**Response:**
```json
{
  "indexed_documents": 150,
  "working_dir": "./lightrag_data",
  "model_name": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "is_initialized": true,
  "available_sources": ["local_files", "google_drive"]
}
```

---

## 에러 응답

모든 API는 에러 발생 시 다음 형식으로 응답합니다:

```json
{
  "detail": "에러 메시지"
}
```

**HTTP 상태 코드:**
| 코드 | 설명 |
|------|------|
| 400 | 잘못된 요청 (파라미터 오류) |
| 500 | 서버 내부 오류 |
| 501 | 지원하지 않는 기능 (의존성 미설치) |

---

## Python 클라이언트 예시

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. 로컬 파일 인덱싱
response = requests.post(f"{BASE_URL}/index/local", json={
    "path": "./rag_raw_pdfs",
    "recursive": True
})
print(response.json())

# 2. RAG 쿼리
response = requests.post(f"{BASE_URL}/query", json={
    "question": "데이터 엔지니어링의 주요 역할은?",
    "mode": "hybrid"
})
result = response.json()
print(f"답변: {result['answer']}")
print(f"처리 시간: {result['processing_time_ms']}ms")

# 3. 시스템 상태 확인
response = requests.get(f"{BASE_URL}/stats")
print(response.json())
```

---

## 환경 변수

`.env` 파일에 설정:

```bash
# 필수
OPENAI_API_KEY=sk-...

# 선택 (Google Drive)
GOOGLE_CREDENTIALS_PATH=./credentials.json
GOOGLE_DRIVE_FOLDER_ID=1ABC...

# 선택 (Slack)
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNELS=C01234567,C89012345

# 선택 (Notion)
NOTION_API_KEY=secret_...
NOTION_DATABASES=abc123,def456
```
