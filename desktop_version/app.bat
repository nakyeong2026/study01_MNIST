@echo off
rem Double-click this file to launch the handwriting digit recognizer (app.py) with no console window.
cd /d "%~dp0"
start "" "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\pythonw.exe" "app.py"
