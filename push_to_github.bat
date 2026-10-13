@echo off
setlocal
title Push AutoValuate AI to GitHub
cd /d "%~dp0"

set "PATH=%USERPROFILE%\AppData\Local\Programs\Git\cmd;%PATH%"

echo =========================================================================
echo       AUTOVALUATE AI - PUSH TO GITHUB FOR PERMANENT DEPLOYMENT
echo =========================================================================
echo.
echo Step 1: Create a new repository on GitHub:
echo         https://github.com/new
echo         (Name it 'autovaluate-ai', do NOT check 'Add README')
echo.
echo Step 2: Copy your GitHub Repository URL (HTTPS)
echo         Example: https://github.com/your-username/autovaluate-ai.git
echo.
set /p REPO_URL="Paste your GitHub Repo URL here: "

if "%REPO_URL%"=="" (
    echo Error: No repository URL provided.
    pause
    exit /b
)

echo.
echo Connecting to GitHub repository...
git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%
git branch -M main

echo Staging and committing files...
git add .
git commit -m "Production AutoValuate AI deployment ready" >nul 2>&1

echo Pushing code to GitHub...
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo =========================================================================
    echo  SUCCESS! Your code and model are now on GitHub!
    echo =========================================================================
    echo.
    echo Now deploy permanently in 1 click:
    echo 1. Go to https://dashboard.render.com
    echo 2. Click "New +" -> "Web Service"
    echo 3. Select your 'autovaluate-ai' repository
    echo 4. Click "Deploy Web Service"
    echo.
    echo Render will build your site and give you a permanent live URL:
    echo https://autovaluate-ai.onrender.com
    echo =========================================================================
) else (
    echo.
    echo Push encountered an error. If prompted for GitHub login, please sign in.
)

pause
