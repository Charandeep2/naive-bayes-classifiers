@echo off
echo Installing Naive Bayes Classifier Dependencies...
echo.

REM Upgrade pip, setuptools, and wheel first
echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Installing project dependencies...
python -m pip install -r requirements.txt

echo.
echo Installation complete!
echo.
echo You can now run the application with: python app.py
pause

