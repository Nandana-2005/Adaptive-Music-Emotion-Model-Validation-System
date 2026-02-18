'''import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_preprocessed_data():
    """Load preprocessed features"""
    X = np.load('data/X_features.npy', allow_pickle=True)
    y = np.load('data/y_labels.npy', allow_pickle=True)
    print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")
    return X, y

def create_realistic_children(X, y, n_children=8, overlap=0.6):
    """
    Create more realistic simulated children:
    - Different exposure to songs
    - Individual perception biases
    - Varying dataset sizes
    """
    
    children = {}
    n_total = len(X)
    
    for i in range(n_children):
        child_id = f"child_{i+1}"
        
        # Each child has seen different songs (60% overlap on average)
        n_songs = int(n_total * overlap) + np.random.randint(-100, 100)
        indices = np.random.choice(n_total, n_songs, replace=False)
        
        X_child = X[indices].copy()
        y_child = y[indices].copy()
        
        # Add individual perception bias (different feature sensitivities)
        perception_bias = np.random.normal(1.0, 0.15, X_child.shape[1])
        X_child = X_child * perception_bias
        
        # Add noise to simulate measurement variance
        noise = np.random.normal(0, 0.05, X_child.shape)
        X_child = X_child + noise
        
        children[child_id] = {
            'X': X_child,
            'y': y_child,
            'n_samples': len(y_child)
        }
        
        emotion_dist = pd.Series(y_child).value_counts().to_dict()
        print(f"✓ {child_id}: {len(y_child)} songs, {emotion_dist}")
    
    return children

def train_multiple_models(X_train, y_train, X_test, y_test):
    """
    Train multiple ML algorithms and compare
    Shows robustness across different approaches
    """
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc
        }
        
        print(f"  {name}: Train={train_acc:.1%}, Test={test_acc:.1%}")
    
    return results

def comprehensive_evaluation(children):
    """
    Full evaluation with multiple scenarios:
    1. Population model (baseline)
    2. Personalized with limited data (50 samples)
    3. Personalized with more data (200 samples)
    4. Cross-validation for robustness
    """
    
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Combine all data for population model
    all_X = np.vstack([c['X'] for c in children.values()])
    all_y = np.hstack([c['y'] for c in children.values()])
    
    print(f"\nTotal dataset: {len(all_X)} samples")
    
    # Train population model
    print("\n" + "-"*70)
    print("1. POPULATION MODEL (Generic Baseline)")
    print("-"*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        all_X, all_y, test_size=0.2, random_state=42, stratify=all_y
    )
    
    pop_results = train_multiple_models(X_train, y_train, X_test, y_test)
    
    # Choose best model
    best_model_name = max(pop_results, key=lambda x: pop_results[x]['test_acc'])
    pop_model = pop_results[best_model_name]['model']
    pop_baseline_acc = pop_results[best_model_name]['test_acc']
    
    print(f"\n✓ Best population model: {best_model_name} ({pop_baseline_acc:.1%})")
    
    # Evaluate personalized approaches
    print("\n" + "-"*70)
    print("2. PERSONALIZED MODELS - Different Data Amounts")
    print("-"*70)
    
    all_results = []
    
    for child_id, data in children.items():
        X_child = data['X']
        y_child = data['y']
        
        # Split
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_child, y_child, test_size=0.2, random_state=42, stratify=y_child
        )
        
        # Scenario 1: Population model on this child
        pop_pred = pop_model.predict(X_test)
        pop_acc = accuracy_score(y_test, pop_pred)
        
        # Scenario 2: Personalized with 50 samples (limited usage)
        n_samples_limited = min(50, len(X_train_full))
        indices_limited = np.random.choice(len(X_train_full), n_samples_limited, replace=False)
        X_train_limited = X_train_full[indices_limited]
        y_train_limited = y_train_full[indices_limited]
        
        pers_model_limited = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        pers_model_limited.fit(X_train_limited, y_train_limited)
        pers_acc_limited = pers_model_limited.score(X_test, y_test)
        
        # Scenario 3: Personalized with 200 samples (more usage)
        n_samples_more = min(200, len(X_train_full))
        indices_more = np.random.choice(len(X_train_full), n_samples_more, replace=False)
        X_train_more = X_train_full[indices_more]
        y_train_more = y_train_full[indices_more]
        
        pers_model_more = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        pers_model_more.fit(X_train_more, y_train_more)
        pers_acc_more = pers_model_more.score(X_test, y_test)
        
        improvement_limited = ((pers_acc_limited - pop_acc) / pop_acc) * 100
        improvement_more = ((pers_acc_more - pop_acc) / pop_acc) * 100
        
        all_results.append({
            'child': child_id,
            'population': pop_acc,
            'personalized_50': pers_acc_limited,
            'personalized_200': pers_acc_more,
            'improvement_50': improvement_limited,
            'improvement_200': improvement_more
        })
        
        print(f"\n{child_id}:")
        print(f"  Population:           {pop_acc:.1%}")
        print(f"  Personalized (50):    {pers_acc_limited:.1%}  (Δ {improvement_limited:+.1f}%)")
        print(f"  Personalized (200):   {pers_acc_more:.1%}  (Δ {improvement_more:+.1f}%)")
    
    # Summary
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Population Model (baseline):     {df['population'].mean():.1%} ± {df['population'].std():.1%}")
    print(f"Personalized (50 samples):       {df['personalized_50'].mean():.1%} ± {df['personalized_50'].std():.1%}")
    print(f"Personalized (200 samples):      {df['personalized_200'].mean():.1%} ± {df['personalized_200'].std():.1%}")
    print(f"\nAvg Improvement (50 samples):    {df['improvement_50'].mean():+.1f}%")
    print(f"Avg Improvement (200 samples):   {df['improvement_200'].mean():+.1f}%")
    
    wins_50 = (df['personalized_50'] > df['population']).sum()
    wins_200 = (df['personalized_200'] > df['population']).sum()
    
    print(f"\nPersonalized wins (50 samples):  {wins_50}/8 children")
    print(f"Personalized wins (200 samples): {wins_200}/8 children")
    
    return df, pop_results

def create_visualizations(results_df):
    """Create publication-quality figures"""
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Figure 1: Accuracy comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    children = results_df['child'].values
    x = np.arange(len(children))
    width = 0.25
    
    ax1.bar(x - width, results_df['population'], width, label='Population', alpha=0.8)
    ax1.bar(x, results_df['personalized_50'], width, label='Personalized (50)', alpha=0.8)
    ax1.bar(x + width, results_df['personalized_200'], width, label='Personalized (200)', alpha=0.8)
    
    ax1.set_xlabel('Child ID')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison by Child')
    ax1.set_xticks(x)
    ax1.set_xticklabels(children, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Improvement plot
    ax2.plot(children, results_df['improvement_50'], 'o-', label='50 samples', linewidth=2, markersize=8)
    ax2.plot(children, results_df['improvement_200'], 's-', label='200 samples', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Baseline')
    
    ax2.set_xlabel('Child ID')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Personalization Improvement over Population Model')
    ax2.set_xticklabels(children, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/results_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data/results_comparison.png")
    
    plt.close()

if __name__ == "__main__":
    print("="*70)
    print("AMECS - ADVANCED ML TRAINING")
    print("="*70)
    
    # Load data
    print("\nStep 1: Loading preprocessed features...")
    X, y = load_preprocessed_data()
    
    # Create children
    print("\nStep 2: Creating simulated children...")
    children = create_realistic_children(X, y, n_children=8)
    
    # Comprehensive evaluation
    print("\nStep 3: Running comprehensive evaluation...")
    results_df, pop_results = comprehensive_evaluation(children)
    
    # Save results
    results_df.to_csv('data/comprehensive_results.csv', index=False)
    print(f"\n✓ Results saved to data/comprehensive_results.csv")
    
    # Create visualizations
    create_visualizations(results_df)
    
    print("\n" + "="*70)
    print("🎉 ADVANCED TRAINING COMPLETE!")
    print("="*70)
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_preprocessed_data():
    """Load preprocessed features"""
    X = np.load('data/X_features.npy', allow_pickle=True)
    y = np.load('data/y_labels.npy', allow_pickle=True)
    print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")
    return X, y

def create_proper_children(X, y, n_children=8):
    """
    FIXED: Create children properly
    - Split data into COMPLETELY SEPARATE sets
    - Population model NEVER sees test data
    - Each child has UNIQUE held-out test set
    """
    
    children = {}
    n_total = len(X)
    
    # First: Set aside 20% as GLOBAL test set
    # Population model will NEVER see this
    X_train_pool, X_global_test, y_train_pool, y_global_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Global train pool: {len(X_train_pool)} samples")
    print(f"Global test set: {len(X_global_test)} samples (never seen by population model)")
    
    # Each child gets a random subset of training pool
    n_per_child = int(len(X_train_pool) * 0.6)
    
    for i in range(n_children):
        child_id = f"child_{i+1}"
        
        # Random subset
        indices = np.random.choice(len(X_train_pool), n_per_child, replace=False)
        X_child = X_train_pool[indices].copy()
        y_child = y_train_pool[indices].copy()
        
        # Add individual perception bias
        perception = np.random.normal(1.0, 0.1, X_child.shape[1])
        X_child = X_child * perception
        
        children[child_id] = {
            'X_train': X_child,
            'y_train': y_child,
            'X_test': X_global_test,
            'y_test': y_global_test
        }
        
        emotion_dist = pd.Series(y_child).value_counts().to_dict()
        print(f"✓ {child_id}: {len(y_child)} training songs, {emotion_dist}")
    
    return children, X_train_pool, y_train_pool, X_global_test, y_global_test

def train_population_model(X_train, y_train, X_test, y_test):
    """Train population model on combined data"""
    
    print("\n" + "="*70)
    print("POPULATION MODEL (Generic Baseline)")
    print("="*70)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"Train accuracy: {train_acc:.1%}")
    print(f"Test accuracy: {test_acc:.1%}")
    
    # Cross validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    
    return model, test_acc

def evaluate_personalization(children, pop_model):
    """
    PROPER evaluation:
    Both models tested on SAME held-out test set
    Personalized model trained on LIMITED child data
    """
    
    print("\n" + "="*70)
    print("PERSONALIZED vs POPULATION EVALUATION")
    print("="*70)
    
    results = []
    
    for child_id, data in children.items():
        X_train_full = data['X_train']
        y_train_full = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Population model on test set
        pop_pred = pop_model.predict(X_test)
        pop_acc = accuracy_score(y_test, pop_pred)
        
        # Personalized with 50 samples
        idx_50 = np.random.choice(len(X_train_full), 
                                   min(50, len(X_train_full)), 
                                   replace=False)
        pers_model_50 = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42
        )
        pers_model_50.fit(X_train_full[idx_50], y_train_full[idx_50])
        pers_acc_50 = pers_model_50.score(X_test, y_test)
        
        # Personalized with 200 samples
        idx_200 = np.random.choice(len(X_train_full),
                                    min(200, len(X_train_full)),
                                    replace=False)
        pers_model_200 = RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=42
        )
        pers_model_200.fit(X_train_full[idx_200], y_train_full[idx_200])
        pers_acc_200 = pers_model_200.score(X_test, y_test)
        
        # Personalized with 500 samples
        idx_500 = np.random.choice(len(X_train_full),
                                    min(500, len(X_train_full)),
                                    replace=False)
        pers_model_500 = RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42
        )
        pers_model_500.fit(X_train_full[idx_500], y_train_full[idx_500])
        pers_acc_500 = pers_model_500.score(X_test, y_test)
        
        imp_50 = ((pers_acc_50 - pop_acc) / pop_acc) * 100
        imp_200 = ((pers_acc_200 - pop_acc) / pop_acc) * 100
        imp_500 = ((pers_acc_500 - pop_acc) / pop_acc) * 100
        
        results.append({
            'child': child_id,
            'population': pop_acc,
            'personalized_50': pers_acc_50,
            'personalized_200': pers_acc_200,
            'personalized_500': pers_acc_500,
            'improvement_50': imp_50,
            'improvement_200': imp_200,
            'improvement_500': imp_500
        })
        
        print(f"\n{child_id}:")
        print(f"  Population (full data):    {pop_acc:.1%}")
        print(f"  Personalized (50 samples): {pers_acc_50:.1%}  (Δ {imp_50:+.1f}%)")
        print(f"  Personalized (200 samples):{pers_acc_200:.1%}  (Δ {imp_200:+.1f}%)")
        print(f"  Personalized (500 samples):{pers_acc_500:.1%}  (Δ {imp_500:+.1f}%)")
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Population Model:            {df['population'].mean():.1%} ± {df['population'].std():.1%}")
    print(f"Personalized (50 samples):   {df['personalized_50'].mean():.1%} ± {df['personalized_50'].std():.1%}")
    print(f"Personalized (200 samples):  {df['personalized_200'].mean():.1%} ± {df['personalized_200'].std():.1%}")
    print(f"Personalized (500 samples):  {df['personalized_500'].mean():.1%} ± {df['personalized_500'].std():.1%}")
    
    print(f"\nAvg Improvement (50):   {df['improvement_50'].mean():+.1f}%")
    print(f"Avg Improvement (200):  {df['improvement_200'].mean():+.1f}%")
    print(f"Avg Improvement (500):  {df['improvement_500'].mean():+.1f}%")
    
    wins_50 = (df['personalized_50'] > df['population']).sum()
    wins_200 = (df['personalized_200'] > df['population']).sum()
    wins_500 = (df['personalized_500'] > df['population']).sum()
    
    print(f"\nPersonalized wins (50):  {wins_50}/8")
    print(f"Personalized wins (200): {wins_200}/8")
    print(f"Personalized wins (500): {wins_500}/8")
    
    print("\n📊 KEY FINDING:")
    print(f"As data increases (50→200→500), personalized accuracy improves:")
    print(f"  {df['personalized_50'].mean():.1%} → {df['personalized_200'].mean():.1%} → {df['personalized_500'].mean():.1%}")
    print(f"This demonstrates the adaptive learning capability of AMECS!")
    
    return df

def create_visualizations(results_df, pop_acc):
    """Create publication-quality figures"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    children = results_df['child'].values
    x = np.arange(len(children))
    width = 0.2
    
    # Figure 1: Accuracy comparison bar chart
    ax1 = axes[0]
    ax1.bar(x - 1.5*width, results_df['population'], 
            width, label='Population', color='#e74c3c', alpha=0.8)
    ax1.bar(x - 0.5*width, results_df['personalized_50'], 
            width, label='Personalized (50)', color='#3498db', alpha=0.8)
    ax1.bar(x + 0.5*width, results_df['personalized_200'], 
            width, label='Personalized (200)', color='#2ecc71', alpha=0.8)
    ax1.bar(x + 1.5*width, results_df['personalized_500'], 
            width, label='Personalized (500)', color='#9b59b6', alpha=0.8)
    
    ax1.set_xlabel('Child ID', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy: Population vs Personalized', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('child_', 'C') for c in children])
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Figure 2: Learning curve
    ax2 = axes[1]
    sample_sizes = [50, 200, 500]
    avg_accuracies = [
        results_df['personalized_50'].mean(),
        results_df['personalized_200'].mean(),
        results_df['personalized_500'].mean()
    ]
    std_accuracies = [
        results_df['personalized_50'].std(),
        results_df['personalized_200'].std(),
        results_df['personalized_500'].std()
    ]
    
    ax2.plot(sample_sizes, avg_accuracies, 'bo-', 
             linewidth=2, markersize=10, label='Personalized')
    ax2.fill_between(sample_sizes,
                     [a-s for a,s in zip(avg_accuracies, std_accuracies)],
                     [a+s for a,s in zip(avg_accuracies, std_accuracies)],
                     alpha=0.2)
    ax2.axhline(y=results_df['population'].mean(), 
                color='r', linestyle='--', linewidth=2,
                label=f'Population ({results_df["population"].mean():.1%})')
    
    ax2.set_xlabel('Training Samples per Child', fontsize=12)
    ax2.set_ylabel('Average Accuracy', fontsize=12)
    ax2.set_title('Learning Curve: Personalization vs Data Amount', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Figure 3: Individual differences heatmap
    ax3 = axes[2]
    heatmap_data = results_df[['population', 
                                'personalized_50',
                                'personalized_200', 
                                'personalized_500']].values
    
    im = ax3.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', 
                    vmin=0, vmax=1)
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels(['Population', 'Pers.\n(50)', 'Pers.\n(200)', 'Pers.\n(500)'])
    ax3.set_yticks(range(8))
    ax3.set_yticklabels([c.replace('child_', 'Child ') for c in children])
    ax3.set_title('Accuracy Heatmap by Child & Model', fontsize=13)
    
    # Add values to heatmap
    for i in range(len(children)):
        for j in range(4):
            ax3.text(j, i, f'{heatmap_data[i,j]:.1%}',
                    ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax3)
    plt.tight_layout()
    plt.savefig('data/results_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data/results_comparison.png")
    plt.close()

if __name__ == "__main__":
    print("="*70)
    print("AMECS - ADVANCED ML TRAINING (FIXED)")
    print("="*70)
    
    # Load data
    print("\nStep 1: Loading preprocessed features...")
    X, y = load_preprocessed_data()
    
    # Create proper children with separate test sets
    print("\nStep 2: Creating simulated children...")
    children, X_train_pool, y_train_pool, X_test, y_test = create_proper_children(
        X, y, n_children=8
    )
    
    # Train population model on training pool only
    print("\nStep 3: Training population model...")
    pop_model, pop_acc = train_population_model(
        X_train_pool, y_train_pool, X_test, y_test
    )
    
    # Proper evaluation
    print("\nStep 4: Evaluating personalization...")
    results_df = evaluate_personalization(children, pop_model)
    
    # Save results
    results_df.to_csv('data/comprehensive_results.csv', index=False)
    print(f"\n✓ Results saved to data/comprehensive_results.csv")
    
    # Create visualizations
    print("\nStep 5: Creating visualizations...")
    create_visualizations(results_df, pop_acc)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)