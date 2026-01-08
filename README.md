# SQL + LLM 회사 챗봇 (Streamlit)

SQL 데이터베이스와 LLM을 함께 사용해 가상 회사 정보를 만들고, 그 데이터를 기반으로 챗봇과 대화하는 Streamlit 앱입니다.

## 준비 사항

- Python 3.9+ 권장
- OpenAI API Key

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
set OPENAI_API_KEY=YOUR_KEY
streamlit run app.py
```

또는 앱 사이드바의 `OpenAI API Key` 입력란에 직접 입력해도 됩니다.

## 주요 기능

### 1) 설정 탭
- 회사 정보를 입력하고 **LLM으로 가상 회사 데이터 생성**  
- 생성 결과는 SQLite DB(`data/company.db`)에 저장
- **DB 크기**(Small/Medium/Large)로 생성 데이터 양 조절
- **생성 프롬프트**(시스템/유저) 미리보기
- **LLM 응답 원문** 확인 가능

> 직원 계정에는 `login_id`/`password`가 포함됩니다.  
> `ADMIN / admin123` 관리자 계정이 반드시 존재하도록 추가됩니다.

### 2) DB 상태 탭
- 테이블별 행 개수 확인
- 각 테이블의 샘플 데이터 확인

### 3) 챗봇 탭
- DB 컨텍스트를 기반으로 질문/응답
- 대화 컨텍스트 유지
- **챗봇 시스템 프롬프트**를 직접 입력 가능
  - `{guardrails}`: 기본 안전 지시사항
  - `{language}`: 현재 언어

#### LLM 에이전트 초안 기능 (옵션)
- 체크박스를 켜면, **의도 → 초안 생성 → 확인 후 전송** 흐름으로 작동
- **에이전트 모델**을 따로 선택 가능
- 초안을 확인/수정 후 **Confirm and send** 버튼으로 서비스 LLM에 전달

## 언어 전환

사이드바의 버튼으로 한국어/영어 전환이 가능합니다.

## 참고

- DB 파일: `data/company.db`
- 모델 목록 및 프롬프트는 필요에 따라 커스터마이징 가능
