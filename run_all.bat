@echo off
REM Chay toan bo FL: 1 server + 10 client, log rieng tung client
REM Cach dung:  run_all.bat [rounds] [local_epochs]
REM Vi du:      run_all.bat 30 1

set ROUNDS=%1
set EPOCHS=%2
if "%ROUNDS%"=="" set ROUNDS=30
if "%EPOCHS%"=="" set EPOCHS=1

if not exist logs mkdir logs

echo Khoi dong server (%ROUNDS% rounds, %EPOCHS% local epoch)...
start "FL-Server" cmd /k python server_iov.py --mode train --rounds %ROUNDS% --local-epochs %EPOCHS%

REM Cho server mo cong truoc khi client ket noi
timeout /t 10 /nobreak >nul

for /l %%i in (0,1,9) do (
    echo Khoi dong client %%i...
    start /min "FL-Client-%%i" cmd /c "python client_iov.py --client-id %%i > logs\client_%%i.log 2>&1"
    timeout /t 2 /nobreak >nul
)

echo.
echo Da khoi dong 1 server + 10 client.
echo - Theo doi tien trinh o cua so "FL-Server"
echo - Log tung client: logs\client_0.log ... logs\client_9.log
echo - Metric tong hop: metrics_iov.csv
