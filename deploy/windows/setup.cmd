@echo off
setlocal
rem Set up this PC to run MemoryScapes. Run once, and again after every "git pull".
rem   deploy\windows\setup.cmd                 everything (first time: ~6 GB of models)
rem   deploy\windows\setup.cmd --build-gsplat  also build gsplat's CUDA kernels (once)
rem   deploy\windows\setup.cmd --skip-models   after a pull: code and the web build only
rem See docs\DEPLOY_WINDOWS.md.
cd /d "%~dp0..\.."
rem The Python launcher (py) if installed, else python (e.g. the Microsoft Store Python).
set "PY=py -3"
where py >nul 2>&1 || set "PY=python"
%PY% deploy\setup.py %*
