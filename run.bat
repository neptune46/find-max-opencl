@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Detect executable
set EXE=build\Release\ocl_find_max.exe
if not exist "%EXE%" set EXE=build\Debug\ocl_find_max.exe
if not exist "%EXE%" set EXE=ocl_find_max.exe
if not exist "%EXE%" (
  echo Executable not found. Build the project first.
  echo For example:
  echo   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  echo   cmake --build build --config Release
  exit /b 1
)

REM Tuning parameters
set WG=256
set GROUPS_MAX=1024

REM Sizes to test (space separated)
set SIZES=1000000 4000000 16777216 33554432 67108864

set OUT=results.csv
echo size,kernel_ms,passes,wg,items_per_thread> "%OUT%"

REM Run from the executable directory so kernels.cl is found next to the exe
for %%I in ("%EXE%") do set EXEDIR=%%~dpI
pushd "%EXEDIR%" >NUL
for %%S in (%SIZES%) do (
  echo Running size %%S ...
  .\ocl_find_max.exe --size %%S --wg %WG% --groups-max %GROUPS_MAX% --quiet --csv >> "%CD%\..\..\results.csv"
)
popd >NUL

echo Done. Results saved to %OUT%
endlocal
