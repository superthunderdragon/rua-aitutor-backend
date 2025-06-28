# 루아 AI코스웨어 - AI튜터 Backend & 교육 분야 Vector DB 구축

이 프로젝트는 루아 AI코스웨어의 AI튜터 기능 구동을 위해 2022 국가 개정 교육과정 등 교육 관련 데이터를 벡터 DB(Chroma)를 활용하여 저장하고 검색하는 백엔드 서비스입니다. Google의 Gemini LLM 및 Google Generative AI 임베딩 모델을 사용하여 교사용/학생용 응답을 생성하며, SlidesGPT API를 통해 프레젠테이션 자료도 생성합니다.

## 주요 기능

- **2022 국가 개정 교육과정 벡터 DB 구축**: PDF 문서 및 기타 텍스트를 청크 단위로 분할한 후, 임베딩을 생성하여 Chroma 벡터스토어에 저장합니다.
- **교사용 AI튜터 응답 생성**: 교육학 전문 교수의 관점에서 질문에 대한 요약 및 형성 평가 문제 JSON을 생성합니다.
- **학생용 AI튜터 응답 생성**: 친절하고 단계적인 설명을 통해 학생 질문에 응답합니다.
- **교사용 의성 지역 특화 프레젠테이션 자료 생성**: PPT/슬라이드 제작 요청에 대해 SlidesGPT API를 호출하여 프레젠테이션 데이터를 생성합니다.


## 프로젝트 구성

- app.py: Quart 기반의 웹 서버로, /teacher 및 /student 엔드포인트를 제공하여 각각 교사용/학생용 응답을 생성합니다.
- vector_db/rag-db.ipynb: 벡터 DB 생성 및 업데이트에 사용되는 Jupyter Notebook입니다.
- cloudbuild.yaml: Google Cloud Platform 내 Cloud Build를 이용해 Docker 이미지 빌드 및 Cloud Run 배포를 위한 설정 파일입니다.
- requirements.txt: 프로젝트에 필요한 모든 Python 패키지 목록입니다.