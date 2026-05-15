@echo off
cd /d "%~dp0"

echo Building Real City (Release Mode)...
cargo build --release --target x86_64-pc-windows-msvc
if %ERRORLEVEL% neq 0 (
    echo Build failed! Check the errors above.
    pause
    exit /b %ERRORLEVEL%
)

echo Build successful! Launching game...
set RUST_BACKTRACE=1
"target\x86_64-pc-windows-msvc\release\real_city.exe"
pause