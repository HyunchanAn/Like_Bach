@echo off
chcp 65001 > nul
title Like Bach - Terminate Studio Processes
echo =====================================================================
echo  Like Bach v4.6 구동 프로세스 강제 종료를 시작합니다.
echo =====================================================================

echo [1/2] 가동 중인 백그라운드 콘솔 창 및 프로세스 트리 종료 중...

rem 구동 스크립트에서 start 명령 시 부여한 창 제목(Window Title)을 기준으로 강제 종료
taskkill /fi "windowtitle eq Like_Bach_Backend_Process*" /t /f >nul 2>&1
taskkill /fi "windowtitle eq Like_Bach_Frontend_Process*" /t /f >nul 2>&1

rem 만약 포트가 살아있을 경우를 대비한 네트워크 포트 기준 추가 점검 및 종료
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /pid %%a /t /f >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do (
    taskkill /pid %%a /t /f >nul 2>&1
)

echo [2/2] 잔류 가상환경 프로세스 정돈 중...
taskkill /im node.exe /f >nul 2>&1

echo =====================================================================
echo  종료 프로세스가 완료되었습니다. 모든 서버가 정상 차단되었습니다.
echo =====================================================================
timeout /t 3
exit