@echo off
chcp 65001 >nul
title AEGIS-OS v3.0 COMPLETE RESTORED - NASA Space Challenge

echo.
echo ============================================================
echo 🚀 AEGIS-OS v3.0 COMPLETE RESTORED
echo    النظام الأصلي الكامل 3200+ سطر + المساعد الذكي المحلي
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
echo 📦 تثبيت المتطلبات الكاملة...
py -m pip install streamlit pandas numpy plotly scikit-learn requests folium networkx
if errorlevel 1 (
    echo ⚠️ تحذير: بعض المكتبات قد لا تكون مثبتة
    echo 💡 المحاولة مع pip فقط...
    pip install streamlit pandas numpy plotly scikit-learn requests folium networkx
)

echo.
echo 🤖 فحص Ollama (اختياري)...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Ollama غير مثبت - سيعمل المساعد الذكي في الوضع الاحتياطي
    echo 💡 لتثبيت Ollama: https://ollama.ai/download
) else (
    echo ✅ Ollama متوفر - المساعد الذكي سيعمل بكامل القدرات
    echo 📥 تحميل Code Llama...
    ollama pull codellama:latest
)

echo.
echo ============================================================
echo 🎯 النظام المستعاد يحتوي على:
echo ✅ الكود الأصلي الكامل 3200+ سطر
echo ✅ جميع الكلاسات العلمية (SGP4, NRLMSISE-00, Monte Carlo)
echo ✅ DataFusionEngine مع RealTimeTLEIntegrator
echo ✅ 500 قطعة حطام حقيقية مع بيانات فيزيائية
echo ✅ نماذج AI متقدمة (RandomForest, GradientBoosting, K-Means)
echo ✅ 10 تبويبات كاملة بمحتوى علمي حقيقي
echo ✅ رسوم بيانية تفاعلية ثلاثية الأبعاد
echo ✅ المساعد الذكي المحلي مع Ollama
echo ✅ حسابات فيزيائية دقيقة وليس أرقام وهمية
echo ============================================================
echo.

echo 🚀 تشغيل AEGIS-OS v3.0 COMPLETE...
echo 🌐 سيفتح في المتصفح تلقائياً على: http://localhost:8501
echo.

streamlit run AEGIS_COMPLETE_RESTORED.py --server.port 8501 --server.address localhost

pause
