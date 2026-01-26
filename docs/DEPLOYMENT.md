# 배포 가이드

LightRAG API 서버의 Docker 및 Kubernetes 배포 가이드

## Docker 배포

### 사전 준비

1. Docker 및 Docker Compose 설치
2. `.env` 파일 생성 (OpenAI API Key 필수)

```bash
# .env 파일 생성
echo "OPENAI_API_KEY=sk-your-api-key" > .env
```

### 빠른 시작 (Docker Compose)

```bash
# 이미지 빌드 및 실행
docker compose up -d

# 로그 확인
docker compose logs -f

# 서비스 중지
docker compose down
```

### 수동 Docker 빌드 및 실행

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

# 상태 확인
docker ps
curl http://localhost:8000/health
```

### PDF 파일 추가

```bash
# 호스트의 PDF 파일을 마운트된 디렉토리에 복사
cp my-document.pdf ./rag_raw_pdfs/

# 인덱싱 실행
curl -X POST "http://localhost:8000/index/local" \
  -H "Content-Type: application/json" \
  -d '{"path": "/app/rag_raw_pdfs", "recursive": true}'
```

### 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 (필수) | - |
| `OPENAI_MODEL` | 사용할 LLM 모델 | gpt-4o-mini |
| `EMBEDDING_MODEL` | 임베딩 모델 | text-embedding-3-small |
| `CHUNK_SIZE` | 문서 청크 크기 | 1000 |
| `CHUNK_OVERLAP` | 청크 오버랩 | 200 |

### 데이터 영속화

Docker Compose는 `lightrag_data` named volume을 사용하여 인덱스 데이터를 영속화합니다.

```bash
# 볼륨 확인
docker volume ls

# 볼륨 삭제 (주의: 모든 인덱스 데이터 삭제)
docker compose down -v
```

---

## Kubernetes 배포

### 사전 준비

1. Kubernetes 클러스터 (로컬: minikube, kind / 클라우드: EKS, GKE, AKS)
2. kubectl 설치 및 클러스터 연결
3. Docker 이미지 레지스트리 접근

### 이미지 준비

```bash
# 이미지 빌드
docker build -t lightrag-api:latest .

# 레지스트리에 푸시 (예: Docker Hub)
docker tag lightrag-api:latest your-registry/lightrag-api:latest
docker push your-registry/lightrag-api:latest
```

### Secret 설정

```bash
# Secret 파일 수정 (실제 API 키로 변경)
vi k8s/secret.yaml

# 또는 kubectl로 직접 생성
kubectl create namespace lightrag
kubectl -n lightrag create secret generic lightrag-secret \
  --from-literal=OPENAI_API_KEY=sk-your-api-key
```

### 배포

```bash
# Kustomize로 배포
kubectl apply -k k8s/

# 또는 개별 파일로 배포
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### 배포 확인

```bash
# 리소스 상태 확인
kubectl -n lightrag get all

# Pod 로그 확인
kubectl -n lightrag logs -f deployment/lightrag-api

# 상세 상태 확인
kubectl -n lightrag describe pod -l app=lightrag-api
```

### 서비스 접근

#### 방법 1: Port Forward (개발/테스트용)

```bash
kubectl -n lightrag port-forward svc/lightrag-api 8000:80

# 다른 터미널에서
curl http://localhost:8000/health
```

#### 방법 2: NodePort

```bash
# NodePort 서비스 사용 (포트 30800)
curl http://<node-ip>:30800/health
```

#### 방법 3: Ingress

```bash
# Ingress 배포 (nginx ingress controller 필요)
kubectl apply -f k8s/ingress.yaml

# 도메인 설정 후
curl http://lightrag.example.com/health
```

### PDF 파일 업로드

Kubernetes 환경에서 PDF 파일을 PVC에 업로드하는 방법:

```bash
# 임시 Pod 생성하여 파일 복사
kubectl -n lightrag run pdf-uploader --image=busybox --restart=Never -- sleep 3600

# Pod에 PVC 마운트 후 파일 복사
kubectl -n lightrag cp ./my-document.pdf pdf-uploader:/tmp/
kubectl -n lightrag exec pdf-uploader -- cp /tmp/my-document.pdf /app/rag_raw_pdfs/

# 또는 kubectl cp 직접 사용 (PVC가 마운트된 Pod에)
kubectl -n lightrag cp ./my-document.pdf lightrag-api-xxx:/app/rag_raw_pdfs/
```

### 스케일링

```bash
# 주의: LightRAG는 로컬 상태를 유지하므로 단일 레플리카 권장
# 스케일 아웃이 필요한 경우 외부 저장소(Redis, PostgreSQL) 연동 필요
kubectl -n lightrag scale deployment/lightrag-api --replicas=1
```

### 리소스 정리

```bash
# 모든 리소스 삭제
kubectl delete -k k8s/

# 또는 네임스페이스 삭제 (모든 리소스 포함)
kubectl delete namespace lightrag
```

---

## 문제 해결

### Docker

```bash
# 컨테이너 상태 확인
docker ps -a
docker logs lightrag-api

# 컨테이너 내부 접속
docker exec -it lightrag-api /bin/bash

# 헬스체크 실패 시
curl -v http://localhost:8000/health
```

### Kubernetes

```bash
# Pod 상태 확인
kubectl -n lightrag describe pod -l app=lightrag-api

# 이벤트 확인
kubectl -n lightrag get events --sort-by='.lastTimestamp'

# 리소스 사용량 확인
kubectl -n lightrag top pod
```

### 일반적인 문제

1. **OPENAI_API_KEY 누락**
   - `.env` 파일 또는 Secret 확인
   - 로그에서 "OpenAI API key not configured" 메시지 확인

2. **PDF 인덱싱 실패**
   - 볼륨 마운트 확인
   - 파일 권한 확인 (읽기 권한 필요)

3. **메모리 부족**
   - 리소스 limits 증가
   - 청크 크기 조정 (CHUNK_SIZE 감소)

---

## Makefile 명령어

```bash
make help            # 도움말
make docker-build    # Docker 이미지 빌드
make docker-run      # Docker 컨테이너 실행
make docker-stop     # Docker 컨테이너 중지
make docker-logs     # Docker 로그 확인
make k8s-deploy      # Kubernetes 배포
make k8s-delete      # Kubernetes 삭제
make k8s-status      # Kubernetes 상태 확인
make k8s-logs        # Kubernetes 로그 확인
make health          # 헬스체크
make stats           # 시스템 통계
```
