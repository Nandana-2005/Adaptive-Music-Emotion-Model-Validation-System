import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def statistical_significance_tests(results_df):

    
    print("="*70)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*70)
    
    pop = results_df['population'].values
    pers_50 = results_df['personalized_50'].values
    pers_200 = results_df['personalized_200'].values
    
    # Paired t-test (comparing same children)
    print("\n1. PAIRED T-TESTS")
    print("-"*70)
    
    t_stat_50, p_value_50 = stats.ttest_rel(pers_50, pop)
    t_stat_200, p_value_200 = stats.ttest_rel(pers_200, pop)
    
    print(f"Personalized (50) vs Population:")
    print(f"  t-statistic: {t_stat_50:.3f}")
    print(f"  p-value: {p_value_50:.4f}")
    if p_value_50 < 0.05:
        print(f"   Statistically significant (p < 0.05)")
    else:
        print(f"   Not significant at p < 0.05 level")
    
    print(f"\nPersonalized (200) vs Population:")
    print(f"  t-statistic: {t_stat_200:.3f}")
    print(f"  p-value: {p_value_200:.4f}")
    if p_value_200 < 0.05:
        print(f"   Statistically significant (p < 0.05)")
    else:
        print(f"   Not significant at p < 0.05 level")
    
    # Effect size (Cohen's d)
    print("\n2. EFFECT SIZE (Cohen's d)")
    print("-"*70)
    
    def cohens_d(x1, x2):
        pooled_std = np.sqrt(((len(x1)-1)*np.std(x1, ddof=1)**2 + (len(x2)-1)*np.std(x2, ddof=1)**2) / (len(x1)+len(x2)-2))
        return (np.mean(x1) - np.mean(x2)) / pooled_std
    
    d_50 = cohens_d(pers_50, pop)
    d_200 = cohens_d(pers_200, pop)
    
    print(f"Personalized (50) vs Population: d = {d_50:.3f}")
    print(f"Personalized (200) vs Population: d = {d_200:.3f}")
    print(f"\nInterpretation:")
    print(f"  |d| < 0.2: small effect")
    print(f"  |d| < 0.5: medium effect")
    print(f"  |d| >= 0.8: large effect")
    
    # Variance analysis
    print("\n3. VARIANCE ANALYSIS")
    print("-"*70)
    
    print(f"Population model variance: {np.var(pop):.4f}")
    print(f"Personalized (50) variance: {np.var(pers_50):.4f}")
    print(f"Personalized (200) variance: {np.var(pers_200):.4f}")
    print(f"\nNote: Higher variance in personalized models shows")
    print(f"      individual differences are captured")
    
    return {
        'p_value_50': p_value_50,
        'p_value_200': p_value_200,
        'cohens_d_50': d_50,
        'cohens_d_200': d_200
    }

if __name__ == "__main__":
    results = pd.read_csv('data/comprehensive_results.csv')
    stats_results = statistical_significance_tests(results)
    
    # Save stats
    pd.DataFrame([stats_results]).to_csv('data/statistical_tests.csv', index=False)
    print("\n✓ Saved to data/statistical_tests.csv")
