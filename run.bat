@echo off
REM Launches Real City from anywhere. Assets are pinned at compile time via
REM CARGO_MANIFEST_DIR so the binary works regardless of CWD.
set RUST_BACKTRACE=1
"%~dp0target\debug\real_city.exe"
pause
