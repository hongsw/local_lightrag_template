# Korea Tech Education RAG PDF

FastAPI 기반 RAG(Retrieval-Augmented Generation) API 서버. LightRAG 엔진을 사용하여 Knowledge Graph + Vector 검색 지원.

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
├── api/main.py      # FastAPI 앱 진입점
├── rag_engine.py    # LightRAG 래퍼
├── processor.py     # 문서 청킹
└── connectors/      # 데이터 소스 커넥터
    ├── local.py     # 로컬 파일
    ├── gdrive.py    # Google Drive
    ├── slack.py     # Slack
    └── notion.py    # Notion
```

## 주요 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | /query | RAG 쿼리 실행 |
| POST | /index/local | 로컬 파일 인덱싱 |
| GET | /sources | 인덱싱된 소스 목록 |

## 환경 설정

`.env` 파일 필요:
```
OPENAI_API_KEY=sk-...
```

Python 3.12+ 권장
