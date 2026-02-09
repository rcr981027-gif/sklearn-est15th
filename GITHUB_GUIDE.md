# GitHub 연동 및 파일 업로드 가이드

현재 `c:\Users\alsld\github` 폴더에 로컬 Git 저장소는 생성되어 있지만, GitHub 원격 저장소(Remote Repository)와 연결되어 있지 않은 상태입니다.

다음 단계에 따라 GitHub와 연동하고 파일을 업로드할 수 있습니다.

## 1단계: GitHub에서 새 저장소 만들기
1. [GitHub 웹사이트](https://github.com)에 로그인합니다.
2. 우측 상단의 **+** 버튼을 누르고 **New repository**를 클릭합니다.
3. **Repository name**을 입력합니다 (예: `data-science-projects` 또는 `pandas-study`).
4. **Public** (공개) 또는 **Private** (비공개)를 선택합니다.
5. **Create repository** 버튼을 클릭합니다.

## 2단계: 로컬 저장소와 GitHub 연결하기
생성된 저장소 페이지에서 HTTPS 주소(예: `https://github.com/사용자명/저장소명.git`)를 복사한 후, 터미널에서 다음 명령어를 입력합니다.

```bash
# 현재 폴더는 c:\Users\alsld\github\data science\pandas 라고 가정합니다.
# 저장소 루트로 이동 (필요한 경우)
cd ..\..

# 원격 저장소 연결 (주소는 본인의 것으로 변경해야 함)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 연결 확인
git remote -v
```

## 3단계: 파일 업로드 (Commit & Push)
원하는 파일(`credit_card_prediction.ipynb`)을 업로드합니다.

```bash
# 파일이 있는 경로로 이동 (이미 해당 경로라면 생략)
cd "data science\pandas\신용카드 사용자 예측"

# 파일 스테이징 (Git이 추적하도록 추가)
git add credit_card_prediction.ipynb

# 커밋 (변경사항 저장)
git commit -m "Add credit card prediction model"

# GitHub로 푸시 (업로드)
# 처음 푸시할 때는 -u 옵션을 줍니다.
git push -u origin main
# 또는 master 브랜치를 사용 중이라면: git push -u origin master
```

## 4단계: 인증 문제 해결 (로그인 창이 뜰 때)
`git push` 시 로그인 창이 뜨거나 비밀번호를 묻는 경우:
- **브라우저 기반 로그인** 창이 뜨면 그대로 로그인하시면 됩니다.
- **Username/Password**를 묻는 경우, Password란에 GitHub 비밀번호 대신 **Personal Access Token (PAT)**을 입력해야 할 수 있습니다. (2021년 8월부터 비밀번호 인증 지원 종료)

### Personal Access Token 발급 방법:
1. GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
2. Generate new token > repo 권한 체크 > Generate token
3. 발급된 토큰(`ghp_...`)을 복사하여 Password 입력란에 붙여넣기 합니다.
