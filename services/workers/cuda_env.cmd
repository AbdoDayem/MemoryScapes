@echo off
rem Run one command with MSVC and the CUDA toolkit set up, for building gsplat's CUDA
rem extension on Windows (see README "GPU splat training on Windows"). Normal pipeline runs
rem don't need this: the built extension loads without it.
rem
rem   cuda_env.cmd .venv\Scripts\python -m pip install --no-build-isolation ...
rem
rem Finds the newest Visual Studio with the C++ tools (vswhere), and CUDA from CUDA_HOME,
rem else CUDA_PATH (set by the CUDA installer), else the newest under the default folder.
setlocal

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo cuda_env: vswhere.exe not found. Install the Visual Studio 2022 Build Tools with the C++ workload. 1>&2
  exit /b 1
)
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
  echo cuda_env: no Visual Studio with the C++ tools found. Add the "Desktop development with C++" workload. 1>&2
  exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || exit /b 1

if not defined CUDA_HOME if defined CUDA_PATH set "CUDA_HOME=%CUDA_PATH%"
if not defined CUDA_HOME (
  for /f "delims=" %%v in ('dir /b /ad /o:n "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA" 2^>nul') do set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\%%v"
)
if not exist "%CUDA_HOME%\bin\nvcc.exe" (
  echo cuda_env: CUDA toolkit not found. Install the one matching torch's CUDA version, or set CUDA_HOME. 1>&2
  exit /b 1
)
set "PATH=%CUDA_HOME%\bin;%PATH%"
set DISTUTILS_USE_SDK=1
rem Build for this machine's GPU unless told otherwise (8.9 = RTX 40 series).
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=8.9"
if not defined MAX_JOBS set MAX_JOBS=8

%*
