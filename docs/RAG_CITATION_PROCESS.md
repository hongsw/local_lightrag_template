# RAG Citation Verification Process

## Overview

이 시스템은 LightRAG 기반의 Knowledge Graph + Vector RAG 시스템에 자동 인용 검증 기능을 추가한 것입니다.

## 전체 프로세스 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Query with Verification                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 사용자 질문 입력                                                  │
│         ↓                                                           │
│  2. LightRAG Context 추출 (Knowledge Graph + Vector Search)          │
│         ↓                                                           │
│  3. Source 추출 및 점수화 (최대 5개, relevance ≥ 0.4)                  │
│         ↓                                                           │
│  4. LLM 응답 생성 (출처 인용 프롬프트 적용)                             │
│         ↓                                                           │
│  5. 인용 검증 (각 [†N] 인용을 소스와 대조)                              │
│         ↓                                                           │
│  6. 부정확한 인용 제거 및 번호 재정렬                                   │
│         ↓                                                           │
│  7. 정정된 응답 + 검증 로그 반환                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 핵심 컴포넌트

### 1. Source 추출 (`_extract_sources_from_context`)

**입력**: LightRAG에서 반환된 컨텍스트 문자열

**처리**:
- `[Source: 파일명]` 패턴으로 소스 파일 식별
- `[Page N]` 패턴으로 페이지 번호 추출
- 관련성 점수 계산:
  - Position Score (40%): 컨텍스트 내 위치 (앞쪽일수록 높음)
  - Content Score (40%): 추출된 내용 길이
  - Frequency Bonus (20%): 같은 소스 반복 언급 횟수

**출력 제한**:
- 최소 관련성: 0.4
- 최대 소스 수: 5개

### 2. 응답 생성 프롬프트

```
당신은 정확한 인용을 하는 전문 리서처입니다.

## 핵심 원칙 (가장 중요!)
**오직 아래 제공된 출처에 명시적으로 적힌 내용만 답변에 사용하세요.**
- 출처에 없는 내용을 추측하거나 일반 지식으로 보충하지 마세요.
- 출처 내용을 과장하거나 확대 해석하지 마세요.
- 확실하지 않으면 "해당 정보는 제공된 출처에서 확인되지 않습니다"라고 답하세요.

## 인용 규칙
1. 각 문장/주장 끝에 해당 출처 번호를 [†1], [†2] 형식으로 표기하세요.
2. 인용 번호는 해당 내용이 실제로 있는 출처만 사용하세요.
3. 여러 출처가 같은 내용을 다루면 [†1][†2]처럼 병기하세요.

## 제공된 출처:
[†1] 파일명, p.페이지
[†2] 파일명, p.페이지
...
```

### 3. 인용 검증 (`verify_citations`)

**검증 프로세스**:
1. 응답에서 `[†N]` 패턴이 포함된 문장 추출
2. 각 인용에 대해 LLM 기반 검증 수행

**검증 프롬프트**:
```
다음 문장이 출처 내용을 정확하게 인용했는지 검증해주세요.

## 검증할 문장:
"{statement}"

## 출처 내용 (발췌):
"{source_excerpt}"

## 검증 기준:
1. 문장의 핵심 주장이 출처 내용에서 뒷받침되는가?
2. 사실 관계가 왜곡되거나 과장되지 않았는가?
3. 출처에 없는 내용을 추가하지 않았는가?

## 응답 형식 (JSON):
{
    "is_accurate": true/false,
    "confidence": 0.0-1.0,
    "explanation": "검증 결과 설명"
}
```

**검증 결과 분류**:
- **Accurate**: `is_accurate=true` AND `confidence >= 0.7`
- **Inaccurate**: `is_accurate=false` AND `confidence >= 0.7`
- **Uncertain**: `confidence < 0.7`

### 4. 응답 정정 (`query_with_verification`)

**정정 프로세스**:
1. 부정확한 인용 번호 식별
2. 남은 인용 번호 재매핑 (old → new)
3. 응답 텍스트에서 인용 번호 교체/제거
4. References 섹션 재생성

**번호 재정렬 예시**:
```
원본: [†1], [†2], [†3], [†4], [†5]
부정확: [†3], [†5]
결과: [†1], [†2], [†3] (이전 1,2,4가 새로운 1,2,3으로)
```

## 인용 표기 형식

### 본문 인용
- 형식: `[†N]`
- 예시: `미국 제조업은 구조적 위기를 겪고 있습니다[†1].`

### References 섹션
```markdown
### References
- [†1] 파일명.pdf, p.10
- [†2] 파일명.pdf, p.25
```

## API 엔드포인트

### POST /query/verified

**Request**:
```json
{
    "question": "질문 내용",
    "mode": "hybrid",
    "confidence_threshold": 0.7
}
```

**Response**:
```json
{
    "original_answer": "원본 응답",
    "corrected_answer": "정정된 응답",
    "mode": "hybrid",
    "sources": [...],
    "removed_citations": [3, 5],
    "accuracy_rate": 0.75,
    "verification_log": [
        {
            "citation_number": 1,
            "statement": "인용된 문장",
            "source_file": "파일명.pdf",
            "source_page": 10,
            "is_accurate": true,
            "confidence": 0.9,
            "explanation": "검증 설명",
            "status": "accurate"
        }
    ],
    "processing_time_ms": 5000.0
}
```

## 설정 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `max_sources` | 5 | 최대 소스 수 |
| `min_relevance` | 0.4 | 최소 관련성 점수 |
| `confidence_threshold` | 0.7 | 검증 신뢰도 임계값 |
| `min_excerpt_length` | 20 | 최소 발췌문 길이 |

## 품질 보장

### Source 품질 필터
- 관련성 점수 0.4 미만 제외
- 발췌문 20자 미만 제외
- 중복 소스 (같은 파일+페이지) 제거

### 검증 품질 필터
- 문장 최소 5자 이상
- 한글/영문 2자 이상 포함
- 중복 문장+인용 조합 제외

## 대시보드 기능

### Auto-Verify Mode
- 토글로 활성화/비활성화
- 활성화 시 `/query/verified` 엔드포인트 사용
- 정정된 응답 + 검증 로그 자동 표시

### 검증 결과 표시
- 정확/부정확/불확실 아이콘 구분
- 제거된 인용 알림
- 정확도 퍼센트 표시

## 파일 구조

```
src/
├── rag_engine.py          # RAG 엔진 + 검증 로직
├── api/
│   ├── models.py          # Pydantic 모델
│   └── routes/
│       └── query.py       # API 엔드포인트
dashboard.html             # 웹 대시보드
```
