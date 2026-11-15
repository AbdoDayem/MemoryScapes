@echo off
rem Start the site when you log in to Windows (Task Scheduler, task "MemoryScapes").
rem   deploy\windows\autostart.cmd             add the task
rem   deploy\windows\autostart.cmd remove      remove it
rem Extra arguments are passed to start.cmd, e.g.  autostart.cmd --host 0.0.0.0
if /i "%~1"=="remove" (
  schtasks /Delete /TN "MemoryScapes" /F
  exit /b %errorlevel%
)
schtasks /Create /TN "MemoryScapes" /TR "\"%~dp0start.cmd\" %*" /SC ONLOGON /RL LIMITED /F
if errorlevel 1 exit /b 1
echo The site will start when you log in. Start it now with deploy\windows\start.cmd
