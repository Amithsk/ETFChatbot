@echo off
set VENV_PATH=E:\ETFChatbot\etfchatbotenv
set SCRIPT_PATH=E:\ETFChatbot\Code\Prompt\etfreturnPromptReturn.py
set LOG_PATH=E:\ETFChatbot\logs\etfreturnPromptReturn.log

REM Ensure logs folder exists
if not exist "E:\ETFChatbot\logs" mkdir "E:\ETFChatbot\logs"

REM Run the script in background and write output to log
start "" "%VENV_PATH%\Scripts\python.exe" "%SCRIPT_PATH%" > "%LOG_PATH%" 2>&1
