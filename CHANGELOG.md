# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-27

### Added

#### Incremental Indexing
- **IndexTracker**: 파일 해시 기반 중복 인덱싱 방지
- **Auto-skip**: 이미 인덱싱된 파일 자동 건너뛰기
- **Change Detection**: 파일 내용 변경 감지 및 재인덱싱

#### New API Endpoints
- `POST /index/local/sync` - 새 파일만 인덱싱 (증분)
- `GET /index/files` - 인덱싱된 파일 목록 조회
- `POST /index/status` - 파일 인덱싱 상태 확인

#### Enhanced Stats
- `tracked_files` - 추적 중인 파일 수
- `tracked_size_bytes` - 추적 파일 총 용량

### Changed
- `/index/local` 응답에 `documents_skipped` 필드 추가

---

## [0.1.0] - 2025-01-27

### Added

#### Core Features
- **LightRAG Integration**: Knowledge Graph + Vector 검색 엔진 통합
- **Multi-mode Query**: naive, local, global, hybrid 쿼리 모드 지원
- **Document Processing**: PDF, TXT, MD 파일 청킹 및 인덱싱

#### Data Source Connectors
- **Local Files**: 로컬 파일시스템에서 문서 로드
- **Google Drive**: Google Drive 폴더 연동 (선택적)
- **Slack**: Slack 채널 메시지 인덱싱 (선택적)
- **Notion**: Notion 페이지/데이터베이스 연동 (선택적)

#### API Endpoints
- `POST /query` - RAG 쿼리 실행
- `POST /index/local` - 로컬 파일 인덱싱
- `POST /index/google-drive` - Google Drive 인덱싱
- `POST /index/slack` - Slack 채널 인덱싱
- `POST /index/notion` - Notion 페이지 인덱싱
- `DELETE /index/clear` - 인덱스 초기화
- `GET /sources` - 사용 가능한 소스 목록
- `GET /sources/local/files` - 로컬 파일 목록
- `GET /stats` - 시스템 통계
- `GET /health` - 헬스 체크
- `GET /query/modes` - 쿼리 모드 목록

#### Deployment
- **Dockerfile**: Multi-stage 빌드, Python 3.12-slim 기반
- **docker-compose.yml**: 로컬 개발 환경 구성
- **Kubernetes**: 완전한 K8s 매니페스트 세트
  - Namespace, ConfigMap, Secret
  - Deployment with health probes
  - Service (ClusterIP, NodePort)
  - PersistentVolumeClaim
  - Ingress (nginx)
  - Kustomization

#### Documentation
- `README.md` - 프로젝트 개요 및 빠른 시작 가이드
- `CLAUDE.md` - Claude Code 작업 컨텍스트
- `docs/API_USAGE.md` - API 상세 사용법
- `docs/DEPLOYMENT.md` - Docker/K8s 배포 가이드

#### Developer Experience
- `Makefile` - 편의 명령어 (docker, k8s, test, lint)
- `run.py` - CLI 실행 스크립트
- `.gitignore` / `.dockerignore` - 적절한 파일 제외

### Technical Details
- **Framework**: FastAPI 0.100+
- **RAG Engine**: LightRAG (lightrag-hku)
- **Document Loader**: LlamaIndex
- **LLM**: OpenAI GPT-4o-mini (기본)
- **Embedding**: OpenAI text-embedding-3-small (기본)
- **Python**: 3.12+
