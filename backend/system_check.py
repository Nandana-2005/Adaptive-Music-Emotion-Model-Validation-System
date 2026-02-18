import os
import numpy as np
import pandas as pd
import sqlite3

def check_database():
    print("\n" + "="*50)
    print("1. DATABASE CHECK")
    print("="*50)
    
    try:
        conn = sqlite3.connect('amecs.db')
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"✓ Tables found: {[t[0] for t in tables]}")
        
        # Check each table
        for table in ['users', 'interactions', 'feedback']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"✓ {table}: {count} records")
        
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def check_deam_dataset():
    print("\n" + "="*50)
    print("2. DEAM DATASET CHECK")
    print("="*50)
    
    try:
        ann_path = 'data/datasets/DEAM/annotations/annotations/annotations averaged per song/song_level'
        features_path = 'data/datasets/DEAM/features/features'
        
        # Check annotations
        ann_files = os.listdir(ann_path)
        print(f"✓ Annotation files: {ann_files}")
        
        # Check features
        feature_files = os.listdir(features_path)
        print(f"✓ Feature files: {len(feature_files)} songs")
        
        return True
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return False

def check_preprocessed_data():
    print("\n" + "="*50)
    print("3. PREPROCESSED DATA CHECK")
    print("="*50)
    
    try:
        X = np.load('data/X_features.npy', allow_pickle=True)
        y = np.load('data/y_labels.npy', allow_pickle=True)
        
        print(f"✓ X shape: {X.shape}")
        print(f"✓ y shape: {y.shape}")
        print(f"✓ Emotions: {np.unique(y)}")
        print(f"✓ Features per song: {X.shape[1]}")
        
        # Check for NaN/Inf
        if np.isnan(X).any():
            print("⚠️  Warning: NaN values in features")
        else:
            print("✓ No NaN values")
            
        if np.isinf(X).any():
            print("⚠️  Warning: Inf values in features")
        else:
            print("✓ No Inf values")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessed data error: {e}")
        print("  Run: python backend/enhanced_feature_extraction.py")
        return False

def check_ml_results():
    print("\n" + "="*50)
    print("4. ML RESULTS CHECK")
    print("="*50)
    
    try:
        results = pd.read_csv('data/comprehensive_results.csv')
        print(f"✓ Results file found")
        print(f"✓ Children evaluated: {len(results)}")
        print(f"\nResults Summary:")
        print(f"  Avg Population: {results['population'].mean():.1%}")
        print(f"  Avg Personalized (50): {results['personalized_50'].mean():.1%}")
        print(f"  Avg Personalized (200): {results['personalized_200'].mean():.1%}")
        print(f"  Avg Improvement: {results['improvement_%'].mean():+.1f}%")
        return True
    except Exception as e:
        print(f"❌ Results error: {e}")
        print("  Run: python backend/advanced_training.py")
        return False

def check_api():
    print("\n" + "="*50)
    print("5. API CHECK")
    print("="*50)
    
    try:
        import requests
        response = requests.get('http://127.0.0.1:5000/')
        
        if response.status_code == 200:
            print(f"✓ API running: {response.text}")
        else:
            print(f"⚠️  API returned status: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ API not running: {e}")
        print("  Run in separate terminal: python backend/app.py")
        return False

def run_all_checks():
    print("="*50)
    print("AMECS SYSTEM HEALTH CHECK")
    print("="*50)
    
    results = {
        'database': check_database(),
        'dataset': check_deam_dataset(),
        'preprocessed': check_preprocessed_data(),
        'ml_results': check_ml_results(),
        'api': check_api()
    }
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    all_pass = True
    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n🎉 All systems working! Ready for GitHub + Frontend")
    else:
        print("\n⚠️  Fix errors above before proceeding")

if __name__ == "__main__":
    run_all_checks()