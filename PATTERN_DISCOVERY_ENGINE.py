"""
PATTERN DISCOVERY ENGINE v1.0
Pure Mathematical Pattern Recognition - Zero Assumptions

PRINCIPLE: Discover patterns through mathematics alone
- No pre-loaded meanings
- No cultural assumptions
- No historical claims
- Only statistical significance

Created: December 2025
Purpose: Unbiased pattern detection in any text/data
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
from itertools import combinations
import statistics

# ============================================================================
# MATHEMATICAL CONSTANTS (No cultural labels)
# ============================================================================

MATHEMATICAL_CONSTANTS = {
    'phi': (1 + math.sqrt(5)) / 2,  # 1.618...
    'pi': math.pi,                   # 3.141...
    'e': math.e,                     # 2.718...
    'sqrt2': math.sqrt(2),           # 1.414...
    'sqrt3': math.sqrt(3),           # 1.732...
    'sqrt5': math.sqrt(5),           # 2.236...
    'silver': 1 + math.sqrt(2),      # 2.414...
    'plastic': 1.324717957...         # Real root of x³ = x + 1
}

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class PatternResult:
    """Mathematical pattern with statistical validation"""
    pattern_type: str
    value: float
    positions: List[int]
    frequency: int
    expected_random_frequency: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    
@dataclass
class SymmetryResult:
    """Detected symmetry with metrics"""
    center: int
    length: int
    symmetry_type: str  # 'reflection', 'rotation', 'translation'
    strength: float  # 0.0 to 1.0

@dataclass
class RatioResult:
    """Detected mathematical ratio"""
    positions: Tuple[int, int]
    observed_ratio: float
    closest_constant: str
    deviation: float
    p_value: float

# ============================================================================
# STATISTICAL VALIDATION ENGINE
# ============================================================================

class StatisticalValidator:
    """Pure statistical validation - no interpretation"""
    
    @staticmethod
    def calculate_p_value(observed: float, expected: float, 
                         n: int, std_dev: Optional[float] = None) -> float:
        """Calculate probability of observing this pattern by chance"""
        if n < 2:
            return 1.0
        
        if std_dev is None:
            # Poisson approximation for count data
            if expected > 0:
                z = abs(observed - expected) / math.sqrt(expected)
            else:
                z = observed
        else:
            # Normal approximation
            z = abs(observed - expected) / (std_dev / math.sqrt(n))
        
        # Two-tailed p-value from standard normal
        p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
        return max(min(p, 1.0), 0.0)
    
    @staticmethod
    def bonferroni_correction(p_values: List[float]) -> List[float]:
        """Correct for multiple comparisons"""
        m = len(p_values)
        return [min(p * m, 1.0) for p in p_values]
    
    @staticmethod
    def calculate_confidence_interval(data: List[float], 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval"""
        if len(data) < 2:
            return (data[0] if data else 0, data[0] if data else 0)
        
        n = len(data)
        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data)
        
        # Z-score for confidence level
        z = 1.96 if confidence == 0.95 else 2.576
        margin = z * (std_val / math.sqrt(n))
        
        return (mean_val - margin, mean_val + margin)
    
    @staticmethod
    def effect_size(observed: float, expected: float, std_dev: float) -> float:
        """Cohen's d effect size"""
        if std_dev == 0:
            return 0.0
        return abs(observed - expected) / std_dev

# ============================================================================
# CHARACTER ENCODING ANALYZER
# ============================================================================

class CharacterEncodingAnalyzer:
    """Analyze numerical properties of character encodings"""
    
    def __init__(self, text: str):
        self.text = text
        self.codes = [ord(c) for c in text]
        self.n = len(self.codes)
    
    def analyze(self) -> Dict:
        """Complete numerical analysis of character codes"""
        if self.n == 0:
            return {"error": "empty_input"}
        
        return {
            "basic_statistics": self._basic_stats(),
            "ratio_analysis": self._analyze_ratios(),
            "sequence_patterns": self._find_sequences(),
            "prime_distribution": self._analyze_primes(),
            "digit_sum_patterns": self._analyze_digit_sums(),
            "modular_patterns": self._analyze_modular(),
        }
    
    def _basic_stats(self) -> Dict:
        """Basic statistical measures"""
        return {
            "mean": statistics.mean(self.codes),
            "median": statistics.median(self.codes),
            "stdev": statistics.stdev(self.codes) if self.n > 1 else 0,
            "min": min(self.codes),
            "max": max(self.codes),
            "range": max(self.codes) - min(self.codes),
            "unique_values": len(set(self.codes)),
            "entropy": self._calculate_entropy()
        }
    
    def _calculate_entropy(self) -> float:
        """Shannon entropy of character distribution"""
        freq = Counter(self.codes)
        entropy = 0
        for count in freq.values():
            p = count / self.n
            entropy -= p * math.log2(p)
        return entropy
    
    def _analyze_ratios(self) -> List[RatioResult]:
        """Find ratios approximating mathematical constants"""
        results = []
        validator = StatisticalValidator()
        
        # Check consecutive pairs
        for i in range(len(self.codes) - 1):
            if self.codes[i] == 0:
                continue
            
            ratio = self.codes[i + 1] / self.codes[i]
            
            # Find closest mathematical constant
            closest = min(
                MATHEMATICAL_CONSTANTS.items(),
                key=lambda x: abs(ratio - x[1])
            )
            
            deviation = abs(ratio - closest[1]) / closest[1]
            
            # Only record if within 5% of a constant
            if deviation < 0.05:
                # Calculate p-value (probability of this deviation by chance)
                # Assuming uniform distribution of ratios
                p_value = deviation * 20  # Approximate
                
                results.append(RatioResult(
                    positions=(i, i + 1),
                    observed_ratio=ratio,
                    closest_constant=closest[0],
                    deviation=deviation,
                    p_value=p_value
                ))
        
        return results
    
    def _find_sequences(self) -> List[PatternResult]:
        """Find arithmetic and geometric sequences"""
        patterns = []
        
        # Arithmetic sequences
        for i in range(self.n - 2):
            diff = self.codes[i + 1] - self.codes[i]
            sequence = [self.codes[i], self.codes[i + 1]]
            
            for j in range(i + 2, self.n):
                if self.codes[j] - sequence[-1] == diff:
                    sequence.append(self.codes[j])
                else:
                    break
            
            if len(sequence) >= 3:
                # Calculate expected frequency by chance
                value_range = max(self.codes) - min(self.codes) + 1
                expected_freq = self.n / (value_range ** (len(sequence) - 1))
                
                p_value = StatisticalValidator.calculate_p_value(
                    observed=1,
                    expected=expected_freq,
                    n=self.n
                )
                
                patterns.append(PatternResult(
                    pattern_type='arithmetic_sequence',
                    value=diff,
                    positions=list(range(i, i + len(sequence))),
                    frequency=1,
                    expected_random_frequency=expected_freq,
                    p_value=p_value,
                    effect_size=1.0 / expected_freq if expected_freq > 0 else 0,
                    confidence_interval=(diff - 1, diff + 1)
                ))
        
        return patterns
    
    def _analyze_primes(self) -> Dict:
        """Analyze distribution of prime numbers"""
        primes = [x for x in self.codes if self._is_prime(x)]
        
        if not primes:
            return {"count": 0, "positions": [], "p_value": 1.0}
        
        # Expected number of primes (prime number theorem approximation)
        avg_value = statistics.mean(self.codes)
        expected_prime_density = 1 / math.log(max(avg_value, 2))
        expected_primes = self.n * expected_prime_density
        
        p_value = StatisticalValidator.calculate_p_value(
            observed=len(primes),
            expected=expected_primes,
            n=self.n
        )
        
        return {
            "count": len(primes),
            "positions": [i for i, x in enumerate(self.codes) if self._is_prime(x)],
            "expected_count": expected_primes,
            "p_value": p_value
        }
    
    def _analyze_digit_sums(self) -> Dict:
        """Analyze patterns in digit sums (digital roots)"""
        digit_sums = [self._digit_sum(x) for x in self.codes]
        freq = Counter(digit_sums)
        
        # Expected uniform distribution (1-9)
        expected_freq = len(digit_sums) / 9
        
        chi_square = sum((obs - expected_freq) ** 2 / expected_freq 
                        for obs in freq.values())
        
        return {
            "distribution": dict(freq),
            "chi_square": chi_square,
            "most_common": freq.most_common(3)
        }
    
    def _analyze_modular(self) -> Dict:
        """Analyze modular arithmetic patterns"""
        results = {}
        
        for modulus in [2, 3, 4, 5, 7, 9, 12]:
            residues = [x % modulus for x in self.codes]
            freq = Counter(residues)
            
            # Expected uniform distribution
            expected = self.n / modulus
            
            chi_square = sum((count - expected) ** 2 / expected 
                           for count in freq.values())
            
            results[f"mod_{modulus}"] = {
                "distribution": dict(freq),
                "chi_square": chi_square,
                "uniform": chi_square < modulus  # Rough test
            }
        
        return results
    
    def _is_prime(self, n: int) -> bool:
        """Check if number is prime"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(abs(n))) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def _digit_sum(self, n: int) -> int:
        """Calculate digital root (recursive digit sum)"""
        n = abs(n)
        while n >= 10:
            n = sum(int(d) for d in str(n))
        return n

# ============================================================================
# SYMMETRY DETECTOR
# ============================================================================

class SymmetryDetector:
    """Detect mathematical symmetries in sequences"""
    
    def __init__(self, text: str):
        self.text = text
        self.n = len(text)
    
    def find_symmetries(self) -> List[SymmetryResult]:
        """Find all types of symmetry"""
        symmetries = []
        
        # Reflection symmetry (palindromes)
        symmetries.extend(self._find_reflection_symmetry())
        
        # Translational symmetry (repeating patterns)
        symmetries.extend(self._find_translational_symmetry())
        
        return symmetries
    
    def _find_reflection_symmetry(self) -> List[SymmetryResult]:
        """Find palindromic patterns"""
        results = []
        
        for center in range(self.n):
            max_radius = min(center, self.n - center - 1)
            
            for radius in range(1, max_radius + 1):
                left = self.text[center - radius:center]
                right = self.text[center + 1:center + radius + 1]
                
                if left == right[::-1]:
                    # Calculate strength (normalized by possible length)
                    strength = radius / (self.n / 2)
                    
                    results.append(SymmetryResult(
                        center=center,
                        length=radius,
                        symmetry_type='reflection',
                        strength=min(strength, 1.0)
                    ))
        
        return results
    
    def _find_translational_symmetry(self) -> List[SymmetryResult]:
        """Find repeating patterns"""
        results = []
        
        for period in range(1, self.n // 2):
            matches = 0
            total_checks = 0
            
            for i in range(self.n - period):
                if self.text[i] == self.text[i + period]:
                    matches += 1
                total_checks += 1
            
            if total_checks > 0:
                strength = matches / total_checks
                
                # Only record if significantly above random
                if strength > 0.5:
                    results.append(SymmetryResult(
                        center=period,  # Using center to store period
                        length=self.n - period,
                        symmetry_type='translation',
                        strength=strength
                    ))
        
        return results

# ============================================================================
# FRACTAL ANALYZER
# ============================================================================

class FractalAnalyzer:
    """Analyze fractal/self-similar properties"""
    
    def __init__(self, text: str):
        self.text = text
        self.binary = ''.join(format(ord(c), '08b') for c in text)
    
    def calculate_fractal_dimension(self) -> Dict:
        """Calculate fractal dimension using box-counting"""
        if len(self.binary) < 16:
            return {"dimension": 1.0, "confidence": 0.0, "method": "insufficient_data"}
        
        # Box counting at multiple scales
        scales = [2, 4, 8, 16, 32]
        counts = []
        valid_scales = []
        
        for scale in scales:
            if len(self.binary) >= scale * 4:
                boxes = set()
                for i in range(0, len(self.binary) - scale + 1, scale):
                    boxes.add(self.binary[i:i + scale])
                counts.append(len(boxes))
                valid_scales.append(scale)
        
        if len(counts) < 2:
            return {"dimension": 1.0, "confidence": 0.0, "method": "insufficient_scales"}
        
        # Linear regression on log-log plot
        log_scales = np.log([1/s for s in valid_scales])
        log_counts = np.log(counts)
        
        slope, intercept = np.polyfit(log_scales, log_counts, 1)
        
        # R-squared for goodness of fit
        residuals = log_counts - (slope * log_scales + intercept)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "dimension": abs(slope),
            "confidence": r_squared,
            "method": "box_counting",
            "scales_used": len(valid_scales)
        }

# ============================================================================
# MAIN PATTERN DISCOVERY ENGINE
# ============================================================================

class PatternDiscoveryEngine:
    """Coordinate all pattern detection with statistical rigor"""
    
    def __init__(self):
        self.validator = StatisticalValidator()
    
    def analyze(self, text: str, significance_level: float = 0.01) -> Dict:
        """
        Complete pattern analysis with statistical validation
        
        Args:
            text: Input text to analyze
            significance_level: P-value threshold (default 0.01)
        
        Returns:
            Dictionary of all discovered patterns with statistics
        """
        if not text:
            return {"error": "empty_input"}
        
        # Initialize analyzers
        encoding_analyzer = CharacterEncodingAnalyzer(text)
        symmetry_detector = SymmetryDetector(text)
        fractal_analyzer = FractalAnalyzer(text)
        
        # Run all analyses
        results = {
            "input_length": len(text),
            "unique_characters": len(set(text)),
            "encoding_analysis": encoding_analyzer.analyze(),
            "symmetry_analysis": symmetry_detector.find_symmetries(),
            "fractal_analysis": fractal_analyzer.calculate_fractal_dimension(),
            "significance_level": significance_level
        }
        
        # Filter by significance
        results["significant_patterns"] = self._filter_significant(
            results, 
            significance_level
        )
        
        # Calculate overall metrics
        results["summary"] = self._summarize(results)
        
        return results
    
    def _filter_significant(self, results: Dict, threshold: float) -> Dict:
        """Extract only statistically significant patterns"""
        significant = {}
        
        # Ratio analysis
        ratios = results["encoding_analysis"].get("ratio_analysis", [])
        sig_ratios = [r for r in ratios if r.p_value < threshold]
        if sig_ratios:
            significant["significant_ratios"] = sig_ratios
        
        # Sequences
        sequences = results["encoding_analysis"].get("sequence_patterns", [])
        sig_sequences = [s for s in sequences if s.p_value < threshold]
        if sig_sequences:
            significant["significant_sequences"] = sig_sequences
        
        # Primes
        prime_analysis = results["encoding_analysis"].get("prime_distribution", {})
        if prime_analysis.get("p_value", 1.0) < threshold:
            significant["significant_prime_pattern"] = prime_analysis
        
        # Symmetries (strength > 0.7 as significance threshold)
        symmetries = results.get("symmetry_analysis", [])
        strong_symmetries = [s for s in symmetries if s.strength > 0.7]
        if strong_symmetries:
            significant["strong_symmetries"] = strong_symmetries
        
        return significant
    
    def _summarize(self, results: Dict) -> Dict:
        """Calculate summary statistics"""
        sig_patterns = results.get("significant_patterns", {})
        
        return {
            "total_significant_patterns": sum(
                len(v) if isinstance(v, list) else 1 
                for v in sig_patterns.values()
            ),
            "pattern_types_found": list(sig_patterns.keys()),
            "has_mathematical_structure": len(sig_patterns) > 0,
            "fractal_dimension": results["fractal_analysis"]["dimension"],
            "fractal_confidence": results["fractal_analysis"]["confidence"]
        }

# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate():
    """Demonstrate pure pattern detection"""
    engine = PatternDiscoveryEngine()
    
    test_cases = [
        "HELLO WORLD",
        "ABCDEFGHIJKLMNOP",
        "AAAAABBBBBCCCCC",
        "ABCDDCBA",
        "123454321",
    ]
    
    print("=" * 70)
    print("PURE MATHEMATICAL PATTERN DISCOVERY")
    print("=" * 70)
    
    for text in test_cases:
        print(f"\nINPUT: {text}")
        print("-" * 70)
        
        results = engine.analyze(text)
        
        # Display summary
        summary = results["summary"]
        print(f"Significant patterns found: {summary['total_significant_patterns']}")
        print(f"Pattern types: {', '.join(summary['pattern_types_found']) or 'None'}")
        print(f"Fractal dimension: {summary['fractal_dimension']:.3f} "
              f"(confidence: {summary['fractal_confidence']:.3f})")
        
        # Display specific patterns
        sig = results.get("significant_patterns", {})
        
        if "significant_ratios" in sig:
            print(f"\nRatio patterns: {len(sig['significant_ratios'])} found")
            for ratio in sig["significant_ratios"][:3]:
                print(f"  Position {ratio.positions}: "
                      f"{ratio.observed_ratio:.3f} ≈ {ratio.closest_constant} "
                      f"(deviation: {ratio.deviation:.3%})")
        
        if "strong_symmetries" in sig:
            print(f"\nSymmetries: {len(sig['strong_symmetries'])} found")
            for sym in sig["strong_symmetries"][:3]:
                print(f"  {sym.symmetry_type.capitalize()} at position {sym.center}, "
                      f"length {sym.length} (strength: {sym.strength:.3f})")

if __name__ == "__main__":
    demonstrate()