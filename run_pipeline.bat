@echo off
cd /d C:\Users\Rex\github\volleybot
set PYTHONUNBUFFERED=1
"C:\Users\Rex\github\volleybot\.venv\Scripts\python.exe" scripts/run_pipeline_all.py >> outputs\logs\all_pipeline.log 2>> outputs\logs\all_pipeline.err
