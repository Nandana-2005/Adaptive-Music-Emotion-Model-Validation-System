import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def load_comprehensive_features():
    
    
    print("Loading comprehensive audio features...")
    
    features_path = Path('data/datasets/DEAM/features/features')
    annotation_path = 'data/datasets/DEAM/annotations/annotations/annotations averaged per song/song_level'
    
    # Load annotations
    ann1 = pd.read_csv(f'{annotation_path}/static_annotations_averaged_songs_1_2000.csv')
    ann2 = pd.read_csv(f'{annotation_path}/static_annotations_averaged_songs_2000_2058.csv')
    annotations = pd.concat([ann1, ann2], ignore_index=True)
    
    # Add emotions
    def map_emotion(row):
        v = row[' valence_mean']
        a = row[' arousal_mean']
        if a > 5 and v > 5: return 'happy'
        elif a > 5 and v <= 5: return 'angry'
        elif a <= 5 and v > 5: return 'calm'
        else: return 'sad'
    
    annotations['emotion'] = annotations.apply(map_emotion, axis=1)
    
    # Load ALL feature files (with semicolon delimiter!)
    feature_files = sorted(list(features_path.glob('*.csv')))
    
    print(f"Found {len(feature_files)} audio feature files")
    
    all_data = []
    
    for i, file in enumerate(feature_files):
        song_id = int(file.stem)
        
        # Find annotation
        ann_row = annotations[annotations['song_id'] == song_id]
        if len(ann_row) == 0:
            continue
        
        emotion = ann_row['emotion'].values[0]
        
       
        features_df = pd.read_csv(file, sep=';')
        feature_stats = {
            'song_id': song_id,
            'emotion': emotion
        }
        
        
        for col in features_df.columns:
            if col == 'frameTime':
                continue
            
            if features_df[col].dtype in ['float64', 'int64', 'float32']:
                feature_stats[f'{col}_mean'] = features_df[col].mean()
                feature_stats[f'{col}_std'] = features_df[col].std()
                feature_stats[f'{col}_min'] = features_df[col].min()
                feature_stats[f'{col}_max'] = features_df[col].max()
        
        all_data.append(feature_stats)
        
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(feature_files)} songs...")
    
    
    df = pd.DataFrame(all_data)
    
    print(f"\n✓ Loaded {len(df)} songs")
    print(f"✓ Feature dimensions: {df.shape[1] - 2} features")
    print(f"\nEmotion distribution:")
    print(df['emotion'].value_counts())
    
    return df

def prepare_ml_data(df):
    """Prepare features and labels for ML"""
    
    print(f"\nPreparing ML data...")
    
    
    feature_cols = [col for col in df.columns if col not in ['song_id', 'emotion']]
    
    X = df[feature_cols]
    y = df['emotion']
    
    print(f"Features shape: {X.shape}")
    
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n✓ Prepared {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
    
    return X_scaled, y.values, scaler

if __name__ == "__main__":
    df = load_comprehensive_features()
    X, y, scaler = prepare_ml_data(df)
    
    
    np.save('data/X_features.npy', X)
    np.save('data/y_labels.npy', y)
    
    print("\nSaved preprocessed data")
    print("  - data/X_features.npy")
    print("  - data/y_labels.npy")
    print(f"\n Ready for ML training with {X.shape[1]} audio features!")
