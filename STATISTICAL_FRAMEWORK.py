"""
STATISTICAL FRAMEWORK v1.0
Rigorous Mathematical Analysis - Zero Interpretation

PRINCIPLE: Numbers speak for themselves
- Report probabilities, not meanings
- Validate with multiple testing corrections
- Provide confidence intervals
- Flag all assumptions explicitly

Created: December 2025
Purpose: Statistical rigor without interpretation bias
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats as scipy_stats
import warnings

# ============================================================================
# CORE STATISTICAL FUNCTIONS
# ============================================================================

class StatisticalTests:
    """Pure statistical hypothesis testing"""
    
    @staticmethod
    def z_test(observed: float, expected: float, std_dev: float, n: int) -> Dict:
        """Two-tailed z-test"""
        if n < 2 or std_dev == 0:
            return {"p_value": 1.0, "z_score": 0.0, "valid": False}
        
        z = (observed - expected) / (std_dev / math.sqrt(n))
        p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
        
        return {
            "z_score": z,
            "p_value": p,
            "significant_at_0.05": p < 0.05,
            "significant_at_0.01": p < 0.01,
            "valid": True
        }
    
    @staticmethod
    def chi_square_test(observed: List[int], expected: List[float]) -> Dict:
        """Chi-square goodness of fit test"""
        if len(observed) != len(expected):
            return {"p_value": 1.0, "valid": False}
        
        chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
        df = len(observed) - 1
        p = 1 - scipy_stats.chi2.cdf(chi2, df)
        
        return {
            "chi_square": chi2,
            "degrees_of_freedom": df,
            "p_value": p,
            "significant_at_0.05": p < 0.05,
            "valid": True
        }
    
    @staticmethod
    def kolmogorov_smirnov_test(data: List[float], 
                                distribution: str = 'uniform') -> Dict:
        """Test if data fits a distribution"""
        if len(data) < 3:
            return {"p_value": 1.0, "valid": False}
        
        if distribution == 'uniform':
            # Normalize to [0, 1]
            min_val, max_val = min(data), max(data)
            if max_val == min_val:
                return {"p_value": 1.0, "valid": False}
            
            normalized = [(x - min_val) / (max_val - min_val) for x in data]
            statistic, p = scipy_stats.kstest(normalized, 'uniform')
        elif distribution == 'normal':
            statistic, p = scipy_stats.kstest(data, 'norm')
        else:
            return {"p_value": 1.0, "valid": False}
        
        return {
            "statistic": statistic,
            "p_value": p,
            "distribution_tested": distribution,
            "rejects_distribution": p < 0.05,
            "valid": True
        }
    
    @staticmethod
    def runs_test(sequence: List) -> Dict:
        """Test for randomness (runs test)"""
        if len(sequence) < 2:
            return {"p_value": 1.0, "valid": False}
        
        # Count runs
        runs = 1
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                runs += 1
        
        # Expected runs
        n1 = sequence.count(sequence[0])
        n2 = len(sequence) - n1
        
        if n1 == 0 or n2 == 0:
            return {"p_value": 1.0, "valid": False}
        
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
                   ((n1 + n2) ** 2 * (n1 + n2 - 1))
        
        if var_runs == 0:
            return {"p_value": 1.0, "valid": False}
        
        z = (runs - expected_runs) / math.sqrt(var_runs)
        p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
        
        return {
            "observed_runs": runs,
            "expected_runs": expected_runs,
            "z_score": z,
            "p_value": p,
            "random_at_0.05": p > 0.05,
            "valid": True
        }

# ============================================================================
# MULTIPLE TESTING CORRECTIONS
# ============================================================================

class MultipleTestingCorrection:
    """Correct for multiple comparisons"""
    
    @staticmethod
    def bonferroni(p_values: List[float]) -> List[float]:
        """Bonferroni correction (most conservative)"""
        m = len(p_values)
        return [min(p * m, 1.0) for p in p_values]
    
    @staticmethod
    def holm_bonferroni(p_values: List[float]) -> List[float]:
        """Holm-Bonferroni (less conservative)"""
        m = len(p_values)
        sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])
        corrected = [0.0] * m
        
        for rank, (original_idx, p) in enumerate(sorted_p):
            corrected[original_idx] = min(p * (m - rank), 1.0)
        
        # Enforce monotonicity
        sorted_corrected = sorted(zip(range(m), corrected), key=lambda x: p_values[x[0]])
        for i in range(1, m):
            if corrected[sorted_corrected[i][0]] < corrected[sorted_corrected[i-1][0]]:
                corrected[sorted_corrected[i][0]] = corrected[sorted_corrected[i-1][0]]
        
        return corrected
    
    @staticmethod
    def benjamini_hochberg(p_values: List[float], fdr: float = 0.05) -> Tuple[List[float], List[bool]]:
        """Benjamini-Hochberg FDR control"""
        m = len(p_values)
        sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])
        
        corrected = [0.0] * m
        rejected = [False] * m
        
        for rank, (original_idx, p) in enumerate(sorted_p):
            corrected_p = p * m / (rank + 1)
            corrected[original_idx] = min(corrected_p, 1.0)
        
        # Enforce monotonicity
        sorted_corrected = sorted(zip(range(m), corrected), key=lambda x: p_values[x[0]])
        for i in range(m - 2, -1, -1):
            if corrected[sorted_corrected[i][0]] > corrected[sorted_corrected[i+1][0]]:
                corrected[sorted_corrected[i][0]] = corrected[sorted_corrected[i+1][0]]
        
        # Determine rejections
        for i, p in enumerate(corrected):
            rejected[i] = p <= fdr
        
        return corrected, rejected

# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================

class ConfidenceIntervals:
    """Calculate confidence intervals"""
    
    @staticmethod
    def bootstrap(data: List[float], statistic: Callable = np.mean,
                 confidence: float = 0.95, iterations: int = 10000) -> Tuple[float, float]:
        """Bootstrap confidence interval"""
        if len(data) < 2:
            val = statistic(data) if data else 0
            return (val, val)
        
        n = len(data)
        bootstrap_stats = []
        
        for _ in range(iterations):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic(sample))
        
        bootstrap_stats.sort()
        lower_idx = int((1 - confidence) / 2 * iterations)
        upper_idx = int((1 + confidence) / 2 * iterations)
        
        return (bootstrap_stats[lower_idx], bootstrap_stats[upper_idx])
    
    @staticmethod
    def parametric(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Parametric confidence interval (assumes normality)"""
        if len(data) < 2:
            val = np.mean(data) if data else 0
            return (val, val)
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        
        # t-distribution critical value
        t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin = t_crit * std / math.sqrt(n)
        
        return (mean - margin, mean + margin)
    
    @staticmethod
    def proportion(successes: int, trials: int, 
                  confidence: float = 0.95) -> Tuple[float, float]:
        """Wilson score interval for proportions"""
        if trials == 0:
            return (0.0, 0.0)
        
        p = successes / trials
        z = scipy_stats.norm.ppf((1 + confidence) / 2)
        
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
        
        return (max(0, center - margin), min(1, center + margin))

# ============================================================================
# EFFECT SIZE CALCULATIONS
# ============================================================================

class EffectSize:
    """Calculate effect sizes"""
    
    @staticmethod
    def cohens_d(group1: List[float], group2: List[float]) -> float:
        """Cohen's d for two groups"""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    @staticmethod
    def cramers_v(chi_square: float, n: int, rows: int, cols: int) -> float:
        """Cramér's V for categorical association"""
        min_dim = min(rows - 1, cols - 1)
        if n == 0 or min_dim == 0:
            return 0.0
        
        return math.sqrt(chi_square / (n * min_dim))
    
    @staticmethod
    def pearson_r(x: List[float], y: List[float]) -> Dict:
        """Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 3:
            return {"r": 0.0, "p_value": 1.0, "valid": False}
        
        r, p = scipy_stats.pearsonr(x, y)
        
        return {
            "r": r,
            "r_squared": r**2,
            "p_value": p,
            "significant": p < 0.05,
            "valid": True
        }

# ============================================================================
# POWER ANALYSIS
# ============================================================================

class PowerAnalysis:
    """Statistical power calculations"""
    
    @staticmethod
    def power_proportion_test(p1: float, p2: float, n: int, 
                             alpha: float = 0.05) -> float:
        """Power for two-proportion z-test"""
        if n < 2:
            return 0.0
        
        # Pooled proportion
        p_pooled = (p1 + p2) / 2
        
        # Effect size
        effect = abs(p1 - p2)
        
        # Standard errors
        se_null = math.sqrt(2 * p_pooled * (1 - p_pooled) / n)
        se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
        
        if se_null == 0 or se_alt == 0:
            return 0.0
        
        # Critical value
        z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
        
        # Non-centrality parameter
        ncp = effect / se_alt
        
        # Power
        power = 1 - scipy_stats.norm.cdf(z_crit - ncp) + scipy_stats.norm.cdf(-z_crit - ncp)
        
        return max(0.0, min(1.0, power))
    
    @staticmethod
    def required_sample_size(effect_size: float, power: float = 0.8,
                            alpha: float = 0.05) -> int:
        """Required sample size for desired power"""
        if effect_size == 0:
            return float('inf')
        
        z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
        z_beta = scipy_stats.norm.ppf(power)
        
        n = ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(math.ceil(n))

# ============================================================================
# ASSUMPTIONS CHECKER
# ============================================================================

class AssumptionsChecker:
    """Check statistical assumptions"""
    
    @staticmethod
    def check_normality(data: List[float]) -> Dict:
        """Shapiro-Wilk test for normality"""
        if len(data) < 3:
            return {"normal": None, "p_value": 1.0, "valid": False, 
                   "message": "Insufficient data"}
        
        statistic, p = scipy_stats.shapiro(data)
        
        return {
            "test": "Shapiro-Wilk",
            "statistic": statistic,
            "p_value": p,
            "normal": p > 0.05,
            "valid": True
        }
    
    @staticmethod
    def check_equal_variances(groups: List[List[float]]) -> Dict:
        """Levene's test for equal variances"""
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            return {"equal_variances": None, "p_value": 1.0, "valid": False}
        
        statistic, p = scipy_stats.levene(*groups)
        
        return {
            "test": "Levene",
            "statistic": statistic,
            "p_value": p,
            "equal_variances": p > 0.05,
            "valid": True
        }
    
    @staticmethod
    def check_independence(sequence: List) -> Dict:
        """Runs test for independence"""
        return StatisticalTests.runs_test(sequence)

# ============================================================================
# INTEGRATED STATISTICAL REPORT
# ============================================================================

class StatisticalReport:
    """Generate comprehensive statistical report"""
    
    def __init__(self, data: Dict, significance_level: float = 0.05):
        self.data = data
        self.alpha = significance_level
        self.tests = StatisticalTests()
        self.corrections = MultipleTestingCorrection()
        self.ci = ConfidenceIntervals()
    
    def generate(self) -> str:
        """Generate formatted report"""
        report = f"""
{'='*70}
STATISTICAL ANALYSIS REPORT
{'='*70}

SIGNIFICANCE LEVEL: α = {self.alpha}

DATA SUMMARY:
  Sample size: {self.data.get('n', 'N/A')}
  Variables analyzed: {self.data.get('n_variables', 'N/A')}

"""
        
        # Hypothesis tests
        if 'tests' in self.data:
            report += self._format_tests(self.data['tests'])
        
        # Multiple testing corrections
        if 'p_values' in self.data and len(self.data['p_values']) > 1:
            report += self._format_corrections(self.data['p_values'])
        
        # Confidence intervals
        if 'confidence_intervals' in self.data:
            report += self._format_confidence_intervals(self.data['confidence_intervals'])
        
        # Effect sizes
        if 'effect_sizes' in self.data:
            report += self._format_effect_sizes(self.data['effect_sizes'])
        
        # Assumptions
        if 'assumptions' in self.data:
            report += self._format_assumptions(self.data['assumptions'])
        
        report += f"\n{'='*70}\n"
        
        return report
    
    def _format_tests(self, tests: Dict) -> str:
        """Format hypothesis test results"""
        section = "HYPOTHESIS TESTS:\n"
        section += "-" * 70 + "\n"
        
        for test_name, result in tests.items():
            if not result.get('valid', False):
                continue
            
            section += f"\n{test_name.upper()}:\n"
            section += f"  Statistic: {result.get('statistic', result.get('z_score', 'N/A')):.4f}\n"
            section += f"  P-value: {result.get('p_value', 1.0):.4f}\n"
            
            if result.get('p_value', 1.0) < self.alpha:
                section += f"  Result: REJECT null hypothesis (α = {self.alpha})\n"
            else:
                section += f"  Result: FAIL TO REJECT null hypothesis\n"
        
        return section + "\n"
    
    def _format_corrections(self, p_values: List[float]) -> str:
        """Format multiple testing corrections"""
        section = "MULTIPLE TESTING CORRECTIONS:\n"
        section += "-" * 70 + "\n"
        
        bonf = self.corrections.bonferroni(p_values)
        holm = self.corrections.holm_bonferroni(p_values)
        bh, rejected = self.corrections.benjamini_hochberg(p_values, self.alpha)
        
        section += f"  Original p-values: {len(p_values)}\n"
        section += f"  Bonferroni significant: {sum(p < self.alpha for p in bonf)}\n"
        section += f"  Holm-Bonferroni significant: {sum(p < self.alpha for p in holm)}\n"
        section += f"  Benjamini-Hochberg significant: {sum(rejected)}\n\n"
        
        return section
    
    def _format_confidence_intervals(self, intervals: Dict) -> str:
        """Format confidence intervals"""
        section = "CONFIDENCE INTERVALS (95%):\n"
        section += "-" * 70 + "\n"
        
        for param, (lower, upper) in intervals.items():
            section += f"  {param}: [{lower:.4f}, {upper:.4f}]\n"
        
        return section + "\n"
    
    def _format_effect_sizes(self, effects: Dict) -> str:
        """Format effect sizes"""
        section = "EFFECT SIZES:\n"
        section += "-" * 70 + "\n"
        
        for effect_name, value in effects.items():
            section += f"  {effect_name}: {value:.4f}\n"
            
            # Interpret Cohen's d
            if 'cohen' in effect_name.lower():
                if abs(value) < 0.2:
                    interp = "negligible"
                elif abs(value) < 0.5:
                    interp = "small"
                elif abs(value) < 0.8:
                    interp = "medium"
                else:
                    interp = "large"
                section += f"    Interpretation: {interp}\n"
        
        return section + "\n"
    
    def _format_assumptions(self, assumptions: Dict) -> str:
        """Format assumption checks"""
        section = "STATISTICAL ASSUMPTIONS:\n"
        section += "-" * 70 + "\n"
        
        for assumption, result in assumptions.items():
            if not result.get('valid', False):
                continue
            
            section += f"\n{assumption.upper()}:\n"
            section += f"  Test: {result.get('test', 'N/A')}\n"
            section += f"  P-value: {result.get('p_value', 1.0):.4f}\n"
            
            key = result.get('normal') or result.get('equal_variances') or result.get('random_at_0.05')
            if key is not None:
                status = "✓ SATISFIED" if key else "✗ VIOLATED"
                section += f"  Status: {status}\n"
        
        return section + "\n"

# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate():
    """Demonstrate statistical framework"""
    print("="*70)
    print("STATISTICAL FRAMEWORK DEMONSTRATION")
    print("="*70)
    
    # Generate sample data
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 50).tolist()
    uniform_data = np.random.uniform(0, 100, 50).tolist()
    
    # Run tests
    tests = StatisticalTests()
    
    print("\n1. NORMALITY TEST:")
    norm_test = AssumptionsChecker.check_normality(normal_data)
    print(f"   Shapiro-Wilk p-value: {norm_test['p_value']:.4f}")
    print(f"   Data appears normal: {norm_test['normal']}")
    
    print("\n2. DISTRIBUTION FIT TEST:")
    ks_test = tests.kolmogorov_smirnov_test(uniform_data, 'uniform')
    print(f"   KS test p-value: {ks_test['p_value']:.4f}")
    print(f"   Rejects uniform: {ks_test['rejects_distribution']}")
    
    print("\n3. CONFIDENCE INTERVALS:")
    ci = ConfidenceIntervals()
    boot_ci = ci.bootstrap(normal_data)
    param_ci = ci.parametric(normal_data)
    print(f"   Bootstrap 95% CI: [{boot_ci[0]:.2f}, {boot_ci[1]:.2f}]")
    print(f"   Parametric 95% CI: [{param_ci[0]:.2f}, {param_ci[1]:.2f}]")
    
    print("\n4. MULTIPLE TESTING CORRECTION:")
    p_values = [0.001, 0.01, 0.03, 0.08, 0.15]
    corrections = MultipleTestingCorrection()
    bonf = corrections.bonferroni(p_values)
    print(f"   Original p-values: {p_values}")
    print(f"   Bonferroni corrected: {[f'{p:.4f}' for p in bonf]}")
    print(f"   Significant at 0.05: {sum(p < 0.05 for p in bonf)}/{len(p_values)}")

if __name__ == "__main__":
    demonstrate()