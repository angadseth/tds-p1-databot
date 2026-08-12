@echo off
REM Runs the P1 databot watchdog. Wired to a Windows Scheduled Task every 5 min.
cd /d "%~dp0"
"C:\Users\24f20\AppData\Local\Programs\Python\Python314\Scripts\uv.exe" run --with requests python watch_bot.py --alert >> watch.log 2>&1
