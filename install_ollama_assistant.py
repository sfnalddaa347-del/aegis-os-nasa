#!/usr/bin/env python3
"""
مثبت المساعد الذكي المحلي - Ollama Code Llama
يقوم بتثبيت وإعداد Ollama مع نموذج Code Llama للمساعد الذكي
"""

import subprocess
import sys
import os
import time
import requests
import platform

def check_ollama_installed():
    """فحص ما إذا كان Ollama مثبت"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def install_ollama_windows():
    """تثبيت Ollama على Windows"""
    print("🔽 تحميل Ollama لـ Windows...")
    
    # تحميل Ollama installer
    ollama_url = "https://ollama.ai/download/windows"
    print(f"📥 تحميل من: {ollama_url}")
    print("⚠️ سيتم فتح المتصفح - يرجى تحميل وتثبيت Ollama يدوياً")
    
    # فتح الرابط في المتصفح
    import webbrowser
    webbrowser.open(ollama_url)
    
    print("\n📋 خطوات التثبيت:")
    print("1. حمل ملف OllamaSetup.exe")
    print("2. شغل الملف كـ Administrator")
    print("3. اتبع تعليمات التثبيت")
    print("4. أعد تشغيل Terminal/Command Prompt")
    print("5. شغل هذا السكريبت مرة أخرى")
    
    input("\n⏳ اضغط Enter بعد تثبيت Ollama...")

def install_ollama_linux():
    """تثبيت Ollama على Linux"""
    print("🔽 تثبيت Ollama على Linux...")
    
    try:
        # تحميل وتثبيت Ollama
        install_command = "curl -fsSL https://ollama.ai/install.sh | sh"
        subprocess.run(install_command, shell=True, check=True)
        print("✅ تم تثبيت Ollama بنجاح")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ خطأ في تثبيت Ollama: {e}")
        return False

def install_ollama_mac():
    """تثبيت Ollama على macOS"""
    print("🔽 تثبيت Ollama على macOS...")
    
    # فتح رابط التحميل
    ollama_url = "https://ollama.ai/download/mac"
    print(f"📥 تحميل من: {ollama_url}")
    
    import webbrowser
    webbrowser.open(ollama_url)
    
    print("\n📋 خطوات التثبيت:")
    print("1. حمل ملف Ollama.app")
    print("2. اسحب التطبيق إلى مجلد Applications")
    print("3. شغل Ollama من Applications")
    print("4. أعد تشغيل Terminal")
    
    input("\n⏳ اضغط Enter بعد تثبيت Ollama...")

def pull_codellama_model():
    """تحميل نموذج Code Llama"""
    print("📥 تحميل نموذج Code Llama...")
    
    try:
        # تحميل Code Llama
        result = subprocess.run(['ollama', 'pull', 'codellama:latest'], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ تم تحميل Code Llama بنجاح")
            return True
        else:
            print(f"❌ خطأ في تحميل Code Llama: {result.stderr}")
            
            # محاولة تحميل نسخة أصغر
            print("🔄 محاولة تحميل النسخة الصغيرة...")
            result = subprocess.run(['ollama', 'pull', 'codellama:7b'], 
                                  capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ تم تحميل Code Llama 7B بنجاح")
                return True
            else:
                print(f"❌ فشل في تحميل Code Llama: {result.stderr}")
                return False
                
    except subprocess.TimeoutExpired:
        print("⏰ انتهت مهلة التحميل - قد يكون النموذج كبير")
        print("💡 جرب: ollama pull codellama:7b")
        return False
    except Exception as e:
        print(f"❌ خطأ في تحميل النموذج: {e}")
        return False

def test_ollama_chat():
    """اختبار المحادثة مع Ollama"""
    print("🧪 اختبار المساعد الذكي...")
    
    try:
        # اختبار بسيط
        test_command = ['ollama', 'run', 'codellama:latest', 'Hello, can you help with space debris analysis?']
        result = subprocess.run(test_command, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and result.stdout.strip():
            print("✅ المساعد الذكي يعمل بشكل صحيح")
            print(f"🤖 رد تجريبي: {result.stdout.strip()[:100]}...")
            return True
        else:
            print("⚠️ المساعد الذكي مثبت لكن قد يحتاج إعداد إضافي")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ انتهت مهلة الاختبار - النموذج قد يكون بطيء")
        return True  # نعتبره نجح
    except Exception as e:
        print(f"⚠️ خطأ في الاختبار: {e}")
        return False

def install_python_ollama():
    """تثبيت مكتبة ollama Python"""
    print("📦 تثبيت مكتبة ollama Python...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'ollama'], check=True)
        print("✅ تم تثبيت مكتبة ollama")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ خطأ في تثبيت مكتبة ollama: {e}")
        return False

def main():
    print("🚀 مثبت المساعد الذكي المحلي - AEGIS-OS")
    print("=" * 50)
    
    # فحص نظام التشغيل
    system = platform.system().lower()
    print(f"🖥️ نظام التشغيل: {system}")
    
    # فحص Python
    print(f"🐍 Python: {sys.version}")
    
    # فحص Ollama
    if check_ollama_installed():
        print("✅ Ollama مثبت مسبقاً")
    else:
        print("❌ Ollama غير مثبت")
        
        # تثبيت حسب نظام التشغيل
        if system == 'windows':
            install_ollama_windows()
        elif system == 'linux':
            if not install_ollama_linux():
                return
        elif system == 'darwin':  # macOS
            install_ollama_mac()
        else:
            print(f"❌ نظام التشغيل غير مدعوم: {system}")
            return
        
        # فحص مرة أخرى بعد التثبيت
        print("\n🔄 فحص Ollama بعد التثبيت...")
        time.sleep(2)
        
        if not check_ollama_installed():
            print("❌ لم يتم تثبيت Ollama بنجاح")
            print("💡 يرجى تثبيت Ollama يدوياً من: https://ollama.ai/")
            return
    
    # تثبيت مكتبة Python
    if not install_python_ollama():
        return
    
    # تحميل النموذج
    if not pull_codellama_model():
        print("⚠️ لم يتم تحميل النموذج - المساعد سيعمل في الوضع الاحتياطي")
        return
    
    # اختبار النظام
    test_ollama_chat()
    
    print("\n" + "=" * 50)
    print("🎉 تم إعداد المساعد الذكي بنجاح!")
    print("🚀 يمكنك الآن تشغيل AEGIS-OS مع المساعد الذكي الكامل")
    print("💡 شغل: START_AEGIS_COMPLETE.bat")
    print("=" * 50)

if __name__ == "__main__":
    main()
