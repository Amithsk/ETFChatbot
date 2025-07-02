@echo off


set VENV_PATH=E:\ETFChatbot\etfchatbotenv
set SCRIPT_PATH=E:\ETFChatbot\Code\Training\train_lora_adapters.py
set PYTHONPATH=E:\ETFChatbot\Code


echo VENV_PATH = %VENV_PATH%
echo SCRIPT_PATH = %SCRIPT_PATH%

REM (Optional) Set a timestamped log file if you want logs too
set LOG_PATH=E:\ETFChatbot\logs\etfreturnPromptReturn.log

REM Make sure log folder exists
if not exist "E:\ETFChatbot\logs" mkdir "E:\ETFChatbot\logs"


REM Start PowerShell in new window, with PYTHONPATH, real-time log view and log file
start powershell -NoExit -Command "$env:PYTHONPATH='E:\ETFChatbot\Code'; & 'E:\ETFChatbot\etfchatbotenv\Scripts\python.exe' 'E:\ETFChatbot\Code\Training\train_lora_adapters.py' | Tee-Object -FilePath 'E:\ETFChatbot\logs\etfreturnPromptReturn.log'"


