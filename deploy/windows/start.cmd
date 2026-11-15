@echo off
setlocal
rem Run the site: the API and the web app, restarted if either stops. Ctrl+C stops both.
rem   deploy\windows\start.cmd                 http://localhost:3000 on this PC
rem   deploy\windows\start.cmd --host 0.0.0.0  also other machines on the network
rem Logs: logs\api.log and logs\web.log. See docs\DEPLOY_WINDOWS.md.
cd /d "%~dp0..\.."
rem The Python launcher (py) if installed, else python (e.g. the Microsoft Store Python).
set "PY=py -3"
where py >nul 2>&1 || set "PY=python"
%PY% deploy\serve.py %*
