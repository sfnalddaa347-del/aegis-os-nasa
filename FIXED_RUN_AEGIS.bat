@echo off
chcp 65001 >nul
title AEGIS-OS v3.0 COMPLETE - FIXED VERSION

echo.
echo ============================================================
echo 🚀 AEGIS-OS v3.0 COMPLETE - VERSION FIXED
echo    تم إصلاح مشكلة PATH - النظام يعمل الآن!
echo ============================================================
echo.

echo 🔍 فحص Python...
py --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python غير متوفر
    pause
    exit /b 1
)

echo ✅ Python متوفر

echo.
echo 🚀 تشغيل AEGIS-OS v3.0 COMPLETE...
echo 🌐 سيفتح في المتصفح تلقائياً على: http://localhost:8501
echo.
echo 💡 تم إصلاح مشكلة PATH - النظام يعمل الآن!
echo.

py -m streamlit run AEGIS_COMPLETE_RESTORED.py --server.port 8501 --server.address localhost

pause
