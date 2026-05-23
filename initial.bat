@echo off
:: UTF-8 코드 페이지로 변경하여 한글 깨짐 방지
chcp 65001 > nul

title Like Bach - Integrated Generative Engine Studio
echo =====================================================================
echo  Like Bach v4.6 Engine & UI Studio 자동 구동 프로세스를 시작합니다.
echo =====================================================================

rem [Step 1] 백엔드 가상환경 및 의존성 환경 변수 설정
echo [1/3] 백엔드 API 서버(FastAPI) 백그라운드 구동 중...
set PYTHONIOENCODING=utf-8

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

rem 중복 실행 방지 및 추적을 위해 창 제목 고정하여 백그라운드 실행
start /min "Like_Bach_Backend_Process" python src/v4/api.py
timeout /t 3 /nobreak > nul

rem [Step 2] 프론트엔드 Vite 개발 서버 구동 (--no-open 옵션으로 중복 브라우저 방지)
echo [2/3] 프론트엔드 UI 서버(Vite) 구동 중...
cd ui/v4-app

start /min "Like_Bach_Frontend_Process" cmd /c "npm run dev -- --no-open"
timeout /t 3 /nobreak > nul

rem [Step 3] 지정된 프론트엔드 UI 주소로 기본 웹 브라우저 호출
echo [3/3] Like Bach UI 스튜디오 접속 중 (기본 웹 브라우저 호출)...
start http://localhost:5173

echo =====================================================================
echo  구동 완료. 본 창을 닫아도 백그라운드 프로세스는 유지됩니다.
echo  서버를 완전히 종료하려면 stop_studio.bat 파일을 실행하십시오.
echo =====================================================================
pause
exit