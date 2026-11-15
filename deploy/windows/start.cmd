@echo off
rem Run the site: the API and the web app, restarted if either stops. Ctrl+C stops both.
rem   deploy\windows\start.cmd                 http://localhost:3000 on this PC
rem   deploy\windows\start.cmd --host 0.0.0.0  also other machines on the network
rem Logs: logs\api.log and logs\web.log. See docs\DEPLOY_WINDOWS.md.
cd /d "%~dp0..\.."
py -3 deploy\serve.py %*
