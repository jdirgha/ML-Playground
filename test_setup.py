#!/usr/bin/env python3
"""
Test script to verify that the Explainable ML Playground setup is working correctly.
Run this script to check if all dependencies are installed and modules can be imported.
"""

import sys
import importlib.util

def test_import(module_name, package=None):
    """Test if a module can be imported"""
    try:
        if package:
            __import__(f"{package}.{module_name}")
            print(f"✅ {package}.{module_name} imported successfully")
        else:
            __import__(module_name)
            print(f"✅ {module_name} imported successfully")
        return True
    except ImportError as e:
        module_full_name = f"{package}.{module_name}" if package else module_name
        print(f"❌ Failed to import {module_full_name}: {e}")
        return False

def main():
    """Run all tests"""
    print("🔬 Explainable ML Playground - Setup Test")
    print("=" * 50)
    
    # Test core dependencies
    print("\n📦 Testing Core Dependencies:")
    dependencies = [
        "streamlit",
        "pandas", 
        "numpy",
        "sklearn",
        "shap",
        "matplotlib",
        "seaborn",
        "plotly",
        "joblib"
    ]
    
    failed_deps = []
    for dep in dependencies:
        if not test_import(dep):
            failed_deps.append(dep)
    
    # Test custom modules
    print("\n🔧 Testing Custom Modules:")
    custom_modules = [
        ("data_handler", "utils"),
        ("model_trainer", "utils"),
        ("explainer", "utils")
    ]
    
    failed_modules = []
    for module, package in custom_modules:
        if not test_import(module, package):
            failed_modules.append(f"{package}.{module}")
    
    # Test sample data files
    print("\n📊 Testing Sample Data:")
    import os
    sample_files = [
        "sample_data/customer_churn.csv",
        "sample_data/house_prices.csv"
    ]
    
    missing_files = []
    for file_path in sample_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} found")
        else:
            print(f"❌ {file_path} not found")
            missing_files.append(file_path)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SETUP TEST SUMMARY")
    print("=" * 50)
    
    if not failed_deps and not failed_modules and not missing_files:
        print("🎉 All tests passed! Your setup is ready.")
        print("\n🚀 To start the application, run:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some issues were found:")
        
        if failed_deps:
            print(f"\n❌ Missing dependencies: {', '.join(failed_deps)}")
            print("   Fix with: pip install -r requirements.txt")
        
        if failed_modules:
            print(f"\n❌ Missing modules: {', '.join(failed_modules)}")
            print("   Check if all files are in the correct location")
        
        if missing_files:
            print(f"\n❌ Missing files: {', '.join(missing_files)}")
            print("   Check if sample data files exist")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 