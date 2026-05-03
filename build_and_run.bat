@echo off
cd /d "%~dp0"

echo Building Real City...
cargo build
if %ERRORLEVEL% neq 0 (
    echo Build failed! Check the errors above.
    pause
    exit /b %ERRORLEVEL%
)

echo Build successful! Launching game...
set RUST_BACKTRACE=1
"target\debug\real_city.exe"
pause
