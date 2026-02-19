import pandas as pd
import numpy as np
from pathlib import Path

def load_deam_data():
    
    
    
    annotations_path = 'data/datasets/DEAM/annotations/annotations/annotations averaged per song/song_level'
    features_path = 'data/datasets/DEAM/features/features'
    
    print("Loading DEAM dataset...")
    
   
    file1 = f'{annotations_path}/static_annotations_averaged_songs_1_2000.csv'
    file2 = f'{annotations_path}/static_annotations_averaged_songs_2000_2058.csv'
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    
    annotations_df = pd.concat([df1, df2], ignore_index=True)
    
    print(f" Loaded {len(annotations_df)} songs")
    print(f" Columns: {list(annotations_df.columns)}")
    
    
    feature_files = list(Path(features_path).glob('*.csv'))[:100]
    
    features_list = []
    for file in feature_files:
        song_id = file.stem
        df = pd.read_csv(file)
        df['song_id'] = song_id
        features_list.append(df)
    
    features_df = pd.concat(features_list, ignore_index=True)
    print(f"✓ Loaded features for {len(feature_files)} songs")
    
    return annotations_df, features_df

def map_to_emotions(valence, arousal):
    
    if arousal > 5 and valence > 5:
        return 'happy'
    elif arousal > 5 and valence <= 5:
        return 'angry'
    elif arousal <= 5 and valence > 5:
        return 'calm'
    else:
        return 'sad'

def prepare_for_ml(annotations_df):
    
    
   
    print(f"\nAvailable columns: {list(annotations_df.columns)}")
    
    
    valence_col = [c for c in annotations_df.columns if 'valence' in c.lower()]
    arousal_col = [c for c in annotations_df.columns if 'arousal' in c.lower()]
    
    if valence_col and arousal_col:
        v_col = valence_col[0]
        a_col = arousal_col[0]
        
        print(f"Using: {v_col} and {a_col}")
        
        annotations_df['emotion'] = annotations_df.apply(
            lambda row: map_to_emotions(row[v_col], row[a_col]),
            axis=1
        )
        
        print("\n✓ Emotion distribution:")
        print(annotations_df['emotion'].value_counts())
    
    return annotations_df

if __name__ == "__main__":
    annotations, features = load_deam_data()
    annotations = prepare_for_ml(annotations)
    
    print(f"\n Dataset ready!")
    print(f"Total songs: {len(annotations)}")
    print(f"Features loaded: {len(features)} samples")
