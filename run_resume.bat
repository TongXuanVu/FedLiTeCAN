@echo off
REM Resume tu checkpoint: run_resume.bat <checkpoint> [rounds] [local_epochs]
REM Vi du: run_resume.bat checkpoints_iov\round_018.pth 30 1

if "%1"=="" (
    echo Thieu checkpoint. Vi du: run_resume.bat checkpoints_iov\round_018.pth 30 1
    exit /b 1
)
set CKPT=%1
set ROUNDS=%2
set EPOCHS=%3
if "%ROUNDS%"=="" set ROUNDS=30
if "%EPOCHS%"=="" set EPOCHS=1

if not exist logs mkdir logs

start "FL-Server" cmd /k python server_iov.py --mode resume --checkpoint %CKPT% --rounds %ROUNDS% --local-epochs %EPOCHS%
timeout /t 10 /nobreak >nul

for /l %%i in (0,1,9) do (
    start /min "FL-Client-%%i" cmd /c "python client_iov.py --client-id %%i > logs\client_%%i.log 2>&1"
    timeout /t 2 /nobreak >nul
)
echo Da khoi dong resume tu %CKPT%.
