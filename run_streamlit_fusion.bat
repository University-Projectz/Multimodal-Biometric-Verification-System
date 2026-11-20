@echo off
setlocal
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
streamlit run src\apps\fusion_streamlit_app.py
