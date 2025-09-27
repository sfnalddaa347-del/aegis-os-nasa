@echo off
chcp 65001 >nul
title AEGIS-OS v3.0 COMPLETE - NASA Space Challenge

echo.
echo ============================================================
echo 🚀 AEGIS-OS v3.0 COMPLETE RESTORED
echo    النظام الأصلي الكامل 3200+ سطر + المساعد الذكي المحلي
echo ============================================================
echo.

echo 🔍 فحص النظام...
py --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python غير متوفر
    pause
    exit /b 1
)

echo ✅ Python متوفر

echo.
echo 📦 تثبيت المتطلبات الأساسية...
py -m pip install streamlit pandas numpy plotly scikit-learn requests folium networkx --quiet
if errorlevel 1 (
    echo ⚠️ تحذير: بعض المكتبات قد تحتاج تثبيت يدوي
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
echo 💡 نصائح للاستخدام:
echo    - اذهب إلى التبويب "🧠 المساعد الذكي المحلي"
echo    - اسأل: "ما هي مخاطر التصادم الحالية؟"
echo    - جرب الرسوم البيانية ثلاثية الأبعاد في التبويب الثاني
echo    - استكشف جميع التبويبات العشرة
echo.

streamlit run AEGIS_COMPLETE_RESTORED.py --server.port 8501 --server.address localhost

pause
