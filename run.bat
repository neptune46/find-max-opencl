@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Resolve script directory (trailing backslash retained)
set SCRIPT_DIR=%~dp0

REM Detect executable relative to script dir
set EXE=%SCRIPT_DIR%build\Release\ocl_find_max.exe
if not exist "%EXE%" set EXE=%SCRIPT_DIR%build\Debug\ocl_find_max.exe
if not exist "%EXE%" set EXE=%SCRIPT_DIR%ocl_find_max.exe
if not exist "%EXE%" (
  echo Executable not found. Build the project first.
  echo For example:
  echo   cmake -S %SCRIPT_DIR% -B %SCRIPT_DIR%build -DCMAKE_BUILD_TYPE=Release
  echo   cmake --build %SCRIPT_DIR%build --config Release
  exit /b 1
)

REM Tuning parameters
set WG=256
set GROUPS_MAX=1024

REM Sizes to test (space separated)
set SIZES=1000000 4000000 16777216 33554432 67108864

set OUT=results.csv
set OUTPATH=%SCRIPT_DIR%%OUT%
echo size,variant,kernel_ms,passes,wg,items_per_thread> "%OUTPATH%"

REM Run from the executable directory so kernels.cl is found next to the exe
for %%I in ("%EXE%") do set EXEDIR=%%~dpI
pushd "%EXEDIR%" >NUL
for %%S in (%SIZES%) do (
  echo Running local ^(CL1.2^) size %%S ...
  .\ocl_find_max.exe --size %%S --wg %WG% --groups-max %GROUPS_MAX% --quiet --csv --variant local >> "%OUTPATH%"
  echo Running wg ^(CL2.0^) size %%S ...
  .\ocl_find_max.exe --size %%S --wg %WG% --groups-max %GROUPS_MAX% --quiet --csv --variant wg >> "%OUTPATH%"
)
popd >NUL

echo Done. Results saved to %OUTPATH%
endlocal
