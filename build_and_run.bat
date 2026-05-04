@echo off
cd /d "%~dp0"

echo Building Real City...
cargo build --target x86_64-pc-windows-msvc
if %ERRORLEVEL% neq 0 (
    echo Build failed! Check the errors above.
    pause
    exit /b %ERRORLEVEL%
)

echo Build successful! Launching game...
set RUST_BACKTRACE=1
"target\x86_64-pc-windows-msvc\debug\real_city.exe"
pause
