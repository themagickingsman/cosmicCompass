"""
INTEGRATED PATTERN DISCOVERY SYSTEM v1.0
Unified Text + Binary + Statistical Analysis

OPTIMIZATION STRATEGY:
1. Analyze text directly (character patterns)
2. Convert to binary and analyze (bit patterns)
3. Cross-validate patterns between both spaces
4. Statistical validation on both streams
5. Synthesize findings with confidence scores

Created: December 2025
Purpose: Maximum pattern discovery through multi-space analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json

# Import our modules
from PATTERN_DISCOVERY_ENGINE import (
    PatternDiscoveryEngine,
    CharacterEncodingAnalyzer,
    SymmetryDetector,
    FractalAnalyzer
)

from BINARY_TRANSLATION_ENGINE import (
    BinaryTranslationEngine,
    BinaryPatternAnalyzer,
    SelfConsistencyValidator
)

from STATISTICAL_FRAMEWORK import (
    StatisticalTests,
    MultipleTestingCorrection,
    ConfidenceIntervals,
    EffectSize
)

# ============================================================================
# CROSS-VALIDATION SYSTEM
# ============================================================================

@dataclass
class CrossValidatedPattern:
    """Pattern validated across multiple analysis spaces"""
    pattern_id: str
    text_space_evidence: Optional[Dict]
    binary_space_evidence: Optional[Dict]
    cross_validation_score: float  # 0.0 to 1.0
    confidence_level: str  # 'high', 'medium', 'low'
    statistical_significance: float  # p-value
    effect_size: float
    
    def is_validated(self) -> bool:
        """Pattern appears in both spaces"""
        return (self.text_space_evidence is not None and 
                self.binary_space_evidence is not None and
                self.cross_validation_score > 0.6)

class CrossSpaceValidator:
    """Validate patterns across text and binary spaces"""
    
    def __init__(self):
        self.stat_tests = StatisticalTests()
        self.effect_calc = EffectSize()
    
    def cross_validate(self, text_patterns: Dict, binary_patterns: Dict) -> List[CrossValidatedPattern]:
        """
        Find patterns that appear in BOTH text and binary analysis
        These are the most reliable patterns
        """
        validated = []
        
        # Check ratio patterns
        text_ratios = text_patterns.get('encoding_analysis', {}).get('ratio_analysis', [])
        binary_ratios = binary_patterns.get('patterns', {}).get('top_patterns', [])
        
        for text_ratio in text_ratios:
            # Look for corresponding binary pattern
            binary_match = self._find_binary_match(text_ratio, binary_ratios)
            
            if binary_match:
                cross_val_score = self._calculate_cross_validation_score(
                    text_ratio, binary_match
                )
                
                validated.append(CrossValidatedPattern(
                    pattern_id=f"ratio_{text_ratio.closest_constant}",
                    text_space_evidence={'ratio': text_ratio},
                    binary_space_evidence={'pattern': binary_match},
                    cross_validation_score=cross_val_score,
                    confidence_level=self._classify_confidence(cross_val_score),
                    statistical_significance=text_ratio.p_value,
                    effect_size=1.0 / (text_ratio.deviation + 0.01)
                ))
        
        # Check symmetry patterns
        text_symmetries = text_patterns.get('symmetry_analysis', [])
        binary_palindromes = binary_patterns.get('structure', {}).get('palindromes', 0)
        
        if text_symmetries and binary_palindromes > 0:
            avg_symmetry_strength = np.mean([s.strength for s in text_symmetries])
            
            validated.append(CrossValidatedPattern(
                pattern_id="symmetry_reflection",
                text_space_evidence={'symmetries': len(text_symmetries), 'avg_strength': avg_symmetry_strength},
                binary_space_evidence={'palindromes': binary_palindromes},
                cross_validation_score=min(avg_symmetry_strength + 0.2, 1.0),
                confidence_level='high' if avg_symmetry_strength > 0.7 else 'medium',
                statistical_significance=1.0 - avg_symmetry_strength,
                effect_size=avg_symmetry_strength
            ))
        
        # Check sequence patterns
        text_sequences = text_patterns.get('encoding_analysis', {}).get('sequence_patterns', [])
        binary_pattern_count = binary_patterns.get('patterns', {}).get('discovered', 0)
        
        if text_sequences and binary_pattern_count > 0:
            # Sequences in text space that have binary counterparts
            for seq in text_sequences[:5]:  # Top 5
                if seq.p_value < 0.05:  # Only significant ones
                    validated.append(CrossValidatedPattern(
                        pattern_id=f"sequence_{seq.pattern_type}_pos_{seq.positions[0]}",
                        text_space_evidence={'sequence': seq},
                        binary_space_evidence={'binary_pattern_count': binary_pattern_count},
                        cross_validation_score=0.7,  # Medium confidence
                        confidence_level='medium',
                        statistical_significance=seq.p_value,
                        effect_size=seq.frequency / max(seq.expected_random_frequency, 0.01)
                    ))
        
        # Check fractal properties (text) vs compression (binary)
        text_fractal = text_patterns.get('fractal_analysis', {})
        binary_compression = binary_patterns.get('structure', {}).get('compression_ratio', 1.0)
        
        if text_fractal.get('confidence', 0) > 0.5:
            # Fractal dimension should correlate with compression
            # High fractal dim (>1.5) → low compression (complex)
            # Low fractal dim (<1.2) → high compression (simple)
            fractal_dim = text_fractal.get('dimension', 1.0)
            
            expected_compression = 2.0 - fractal_dim  # Inverse relationship
            compression_match = 1.0 - abs(binary_compression - expected_compression) / 2.0
            
            if compression_match > 0.5:
                validated.append(CrossValidatedPattern(
                    pattern_id="fractal_compression_coherence",
                    text_space_evidence={'fractal_dimension': fractal_dim, 'confidence': text_fractal['confidence']},
                    binary_space_evidence={'compression_ratio': binary_compression},
                    cross_validation_score=compression_match,
                    confidence_level=self._classify_confidence(compression_match),
                    statistical_significance=1.0 - text_fractal['confidence'],
                    effect_size=compression_match
                ))
        
        return validated
    
    def _find_binary_match(self, text_ratio, binary_patterns: List) -> Optional[Dict]:
        """Find binary pattern that matches text ratio"""
        # This is a heuristic: look for binary patterns with similar frequency
        text_freq = 1  # Ratios appear once
        
        for binary_pattern in binary_patterns:
            if binary_pattern.get('frequency', 0) >= text_freq:
                # Found a candidate
                return binary_pattern
        
        return None
    
    def _calculate_cross_validation_score(self, text_evidence, binary_evidence) -> float:
        """Calculate how well text and binary patterns agree"""
        score = 0.0
        
        # If both exist, base score is 0.5
        score += 0.5
        
        # Bonus for text pattern quality
        if hasattr(text_evidence, 'p_value') and text_evidence.p_value < 0.01:
            score += 0.2
        elif hasattr(text_evidence, 'p_value') and text_evidence.p_value < 0.05:
            score += 0.1
        
        # Bonus for binary pattern frequency
        if isinstance(binary_evidence, dict):
            freq = binary_evidence.get('frequency', 0)
            if freq > 5:
                score += 0.2
            elif freq > 2:
                score += 0.1
        
        return min(score, 1.0)
    
    def _classify_confidence(self, score: float) -> str:
        """Classify confidence level"""
        if score > 0.8:
            return 'high'
        elif score > 0.6:
            return 'medium'
        else:
            return 'low'

# ============================================================================
# INTEGRATED ANALYSIS RESULT
# ============================================================================

@dataclass
class IntegratedAnalysisResult:
    """Complete analysis from all systems"""
    # Input
    input_text: str
    input_length: int
    
    # Text space analysis
    text_patterns: Dict
    text_pattern_count: int
    
    # Binary space analysis
    binary_representation: str
    binary_patterns: Dict
    binary_pattern_count: int
    
    # Cross-validated patterns
    cross_validated: List[CrossValidatedPattern]
    high_confidence_patterns: List[CrossValidatedPattern]
    
    # Statistical summary
    combined_p_values: List[float]
    overall_significance: float
    
    # Validation
    self_consistency_score: float
    information_preservation: float
    
    # Synthesis
    pattern_synthesis: Dict
    confidence_score: float  # Overall analysis confidence
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'input': {
                'text': self.input_text[:100] + '...' if len(self.input_text) > 100 else self.input_text,
                'length': self.input_length
            },
            'text_analysis': {
                'pattern_count': self.text_pattern_count,
                'key_patterns': self._summarize_patterns(self.text_patterns)
            },
            'binary_analysis': {
                'representation_length': len(self.binary_representation),
                'pattern_count': self.binary_pattern_count,
                'key_patterns': self._summarize_binary(self.binary_patterns)
            },
            'cross_validation': {
                'validated_patterns': len(self.cross_validated),
                'high_confidence': len(self.high_confidence_patterns),
                'patterns': [
                    {
                        'id': p.pattern_id,
                        'confidence': p.confidence_level,
                        'score': p.cross_validation_score,
                        'p_value': p.statistical_significance,
                        'validated': p.is_validated()
                    }
                    for p in self.cross_validated
                ]
            },
            'synthesis': self.pattern_synthesis,
            'overall_confidence': self.confidence_score
        }
    
    def _summarize_patterns(self, patterns: Dict) -> Dict:
        """Summarize key text patterns"""
        encoding = patterns.get('encoding_analysis', {})
        return {
            'ratios': len(encoding.get('ratio_analysis', [])),
            'sequences': len(encoding.get('sequence_patterns', [])),
            'symmetries': len(patterns.get('symmetry_analysis', [])),
            'fractal_dimension': patterns.get('fractal_analysis', {}).get('dimension', 0)
        }
    
    def _summarize_binary(self, patterns: Dict) -> Dict:
        """Summarize key binary patterns"""
        return {
            'discovered': patterns.get('patterns', {}).get('discovered', 0),
            'palindromes': patterns.get('structure', {}).get('palindromes', 0),
            'compression': patterns.get('structure', {}).get('compression_ratio', 0)
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict(), indent=indent, default=str)

# ============================================================================
# INTEGRATED PATTERN DISCOVERY SYSTEM
# ============================================================================

class IntegratedPatternSystem:
    """
    Main integrated system combining:
    - Text pattern analysis
    - Binary translation and analysis
    - Cross-space validation
    - Statistical rigor
    """
    
    def __init__(self, significance_level: float = 0.01):
        self.significance_level = significance_level
        
        # Initialize subsystems
        self.pattern_engine = PatternDiscoveryEngine()
        self.binary_engine = BinaryTranslationEngine()
        self.cross_validator = CrossSpaceValidator()
        self.stat_tests = StatisticalTests()
        self.corrections = MultipleTestingCorrection()
    
    def analyze(self, text: str, binary_strategy: str = 'adaptive', 
                verbose: bool = False) -> IntegratedAnalysisResult:
        """
        Complete integrated analysis
        
        Args:
            text: Input text in any script
            binary_strategy: Binary conversion strategy
            verbose: Print progress
        
        Returns:
            IntegratedAnalysisResult with all findings
        """
        if verbose:
            print(f"Integrated analysis of {len(text)} characters...")
        
        # Stage 1: Text space analysis
        if verbose:
            print("  [1/5] Analyzing text space patterns...")
        text_patterns = self.pattern_engine.analyze(text, self.significance_level)
        text_pattern_count = self._count_text_patterns(text_patterns)
        
        # Stage 2: Binary space analysis
        if verbose:
            print(f"  [2/5] Converting to binary ({binary_strategy} strategy)...")
        binary_result = self.binary_engine.translate(text, binary_strategy)
        binary_patterns = binary_result
        binary_pattern_count = binary_result['patterns']['discovered']
        
        # Stage 3: Cross-validation
        if verbose:
            print("  [3/5] Cross-validating patterns...")
        cross_validated = self.cross_validator.cross_validate(
            text_patterns, 
            binary_patterns
        )
        
        high_confidence = [p for p in cross_validated if p.confidence_level == 'high']
        
        # Stage 4: Statistical synthesis
        if verbose:
            print("  [4/5] Synthesizing statistics...")
        
        # Collect all p-values
        p_values = []
        
        # Text space p-values
        for ratio in text_patterns.get('encoding_analysis', {}).get('ratio_analysis', []):
            p_values.append(ratio.p_value)
        for seq in text_patterns.get('encoding_analysis', {}).get('sequence_patterns', []):
            p_values.append(seq.p_value)
        
        # Cross-validated p-values
        for pattern in cross_validated:
            p_values.append(pattern.statistical_significance)
        
        # Apply multiple testing correction
        if p_values:
            corrected_p = self.corrections.holm_bonferroni(p_values)
            overall_sig = np.mean(corrected_p)
        else:
            corrected_p = []
            overall_sig = 1.0
        
        # Stage 5: Validation and synthesis
        if verbose:
            print("  [5/5] Generating synthesis...")
        
        # Self-consistency from binary validation
        self_consistency = binary_result['validation']['information_preservation']['preservation_ratio']
        info_preservation = 1.0 if binary_result['validation']['information_preservation']['information_preserved'] else 0.0
        
        # Pattern synthesis
        synthesis = self._synthesize_patterns(
            text_patterns,
            binary_patterns,
            cross_validated
        )
        
        # Overall confidence score
        confidence = self._calculate_overall_confidence(
            text_pattern_count,
            binary_pattern_count,
            len(cross_validated),
            len(high_confidence),
            overall_sig,
            self_consistency
        )
        
        result = IntegratedAnalysisResult(
            input_text=text,
            input_length=len(text),
            text_patterns=text_patterns,
            text_pattern_count=text_pattern_count,
            binary_representation=binary_result['binary']['sequence'],
            binary_patterns=binary_patterns,
            binary_pattern_count=binary_pattern_count,
            cross_validated=cross_validated,
            high_confidence_patterns=high_confidence,
            combined_p_values=corrected_p,
            overall_significance=overall_sig,
            self_consistency_score=self_consistency,
            information_preservation=info_preservation,
            pattern_synthesis=synthesis,
            confidence_score=confidence
        )
        
        if verbose:
            print(f"  Complete! Confidence: {confidence:.1%}")
        
        return result
    
    def _count_text_patterns(self, patterns: Dict) -> int:
        """Count discovered text patterns"""
        count = 0
        encoding = patterns.get('encoding_analysis', {})
        count += len(encoding.get('ratio_analysis', []))
        count += len(encoding.get('sequence_patterns', []))
        count += len(patterns.get('symmetry_analysis', []))
        return count
    
    def _synthesize_patterns(self, text_patterns: Dict, binary_patterns: Dict,
                            cross_validated: List[CrossValidatedPattern]) -> Dict:
        """Synthesize findings from all analysis streams"""
        synthesis = {
            'pattern_agreement': {},
            'unique_insights': {},
            'recommendations': []
        }
        
        # Pattern agreement analysis
        validated_count = len([p for p in cross_validated if p.is_validated()])
        total_patterns = self._count_text_patterns(text_patterns) + binary_patterns['patterns']['discovered']
        
        if total_patterns > 0:
            agreement_rate = validated_count / total_patterns
        else:
            agreement_rate = 0.0
        
        synthesis['pattern_agreement'] = {
            'cross_validated_count': validated_count,
            'total_pattern_count': total_patterns,
            'agreement_rate': agreement_rate,
            'interpretation': self._interpret_agreement(agreement_rate)
        }
        
        # Unique insights from each space
        synthesis['unique_insights']['text_space'] = self._extract_text_insights(text_patterns)
        synthesis['unique_insights']['binary_space'] = self._extract_binary_insights(binary_patterns)
        
        # Generate recommendations
        if agreement_rate > 0.7:
            synthesis['recommendations'].append("High pattern agreement - findings are robust")
        elif agreement_rate > 0.4:
            synthesis['recommendations'].append("Moderate agreement - patterns present but require verification")
        else:
            synthesis['recommendations'].append("Low agreement - patterns may be space-specific or artifacts")
        
        if len(cross_validated) > 0:
            high_conf = [p for p in cross_validated if p.confidence_level == 'high']
            if high_conf:
                synthesis['recommendations'].append(f"Focus on {len(high_conf)} high-confidence patterns")
        
        return synthesis
    
    def _interpret_agreement(self, rate: float) -> str:
        """Interpret agreement rate"""
        if rate > 0.7:
            return "Strong cross-space coherence"
        elif rate > 0.5:
            return "Good pattern consistency"
        elif rate > 0.3:
            return "Moderate pattern overlap"
        else:
            return "Limited cross-validation"
    
    def _extract_text_insights(self, patterns: Dict) -> Dict:
        """Extract key insights from text analysis"""
        insights = {}
        
        encoding = patterns.get('encoding_analysis', {})
        basic = encoding.get('basic_statistics', {})
        
        insights['entropy'] = basic.get('entropy', 0)
        insights['complexity'] = 'high' if basic.get('entropy', 0) > 3.5 else 'medium' if basic.get('entropy', 0) > 2.0 else 'low'
        
        fractal = patterns.get('fractal_analysis', {})
        if fractal.get('confidence', 0) > 0.5:
            insights['fractal_dimension'] = fractal.get('dimension', 1.0)
            insights['self_similarity'] = 'present' if fractal.get('dimension', 1.0) > 1.2 else 'minimal'
        
        return insights
    
    def _extract_binary_insights(self, patterns: Dict) -> Dict:
        """Extract key insights from binary analysis"""
        insights = {}
        
        insights['bit_balance'] = patterns.get('binary', {}).get('balance', 0.5)
        insights['balanced'] = 0.4 < insights['bit_balance'] < 0.6
        
        compression = patterns.get('structure', {}).get('compression_ratio', 1.0)
        insights['compression_ratio'] = compression
        insights['compressibility'] = 'high' if compression < 0.5 else 'medium' if compression < 0.8 else 'low'
        
        return insights
    
    def _calculate_overall_confidence(self, text_count: int, binary_count: int,
                                     validated_count: int, high_conf_count: int,
                                     overall_sig: float, consistency: float) -> float:
        """Calculate overall analysis confidence"""
        # Weighted components
        weights = {
            'pattern_discovery': 0.25,      # Found patterns at all
            'cross_validation': 0.30,       # Patterns agree across spaces
            'statistical_sig': 0.25,        # Statistically significant
            'self_consistency': 0.20        # Binary conversion is valid
        }
        
        # Pattern discovery score
        pattern_score = min((text_count + binary_count) / 20, 1.0)
        
        # Cross-validation score
        if text_count + binary_count > 0:
            cross_val_score = (validated_count * 2) / (text_count + binary_count)
        else:
            cross_val_score = 0.0
        cross_val_score = min(cross_val_score, 1.0)
        
        # Statistical significance score (inverse of p-value)
        sig_score = 1.0 - min(overall_sig, 1.0)
        
        # Self-consistency score (from binary validation)
        consistency_score = min(consistency, 1.0)
        
        # Weighted average
        confidence = (
            pattern_score * weights['pattern_discovery'] +
            cross_val_score * weights['cross_validation'] +
            sig_score * weights['statistical_sig'] +
            consistency_score * weights['self_consistency']
        )
        
        return confidence
    
    def generate_report(self, result: IntegratedAnalysisResult, 
                       format: str = 'text') -> str:
        """Generate comprehensive report"""
        if format == 'json':
            return result.to_json()
        else:
            return self._generate_text_report(result)
    
    def _generate_text_report(self, result: IntegratedAnalysisResult) -> str:
        """Generate formatted text report"""
        report = f"""
{'='*80}
INTEGRATED PATTERN DISCOVERY REPORT
{'='*80}

INPUT:
  Text: {result.input_text[:80]}{'...' if len(result.input_text) > 80 else ''}
  Length: {result.input_length} characters

{'='*80}
TEXT SPACE ANALYSIS
{'='*80}
  Patterns discovered: {result.text_pattern_count}
  Mathematical ratios: {len(result.text_patterns.get('encoding_analysis', {}).get('ratio_analysis', []))}
  Sequences: {len(result.text_patterns.get('encoding_analysis', {}).get('sequence_patterns', []))}
  Symmetries: {len(result.text_patterns.get('symmetry_analysis', []))}
  Fractal dimension: {result.text_patterns.get('fractal_analysis', {}).get('dimension', 1.0):.3f}

{'='*80}
BINARY SPACE ANALYSIS
{'='*80}
  Binary length: {len(result.binary_representation)} bits
  Patterns discovered: {result.binary_pattern_count}
  Bit balance: {result.binary_patterns.get('binary', {}).get('balance', 0.5):.3f}
  Palindromes: {result.binary_patterns.get('structure', {}).get('palindromes', 0)}
  Compression ratio: {result.binary_patterns.get('structure', {}).get('compression_ratio', 1.0):.3f}
  Complexity: {result.binary_patterns.get('structure', {}).get('complexity_class', 'unknown')}

{'='*80}
CROSS-VALIDATION
{'='*80}
  Cross-validated patterns: {len(result.cross_validated)}
  High confidence: {len(result.high_confidence_patterns)}
  Agreement rate: {result.pattern_synthesis['pattern_agreement']['agreement_rate']:.1%}
  Interpretation: {result.pattern_synthesis['pattern_agreement']['interpretation']}

"""
        
        if result.high_confidence_patterns:
            report += f"\nHIGH CONFIDENCE PATTERNS:\n"
            report += f"{'-'*80}\n"
            for i, pattern in enumerate(result.high_confidence_patterns[:5], 1):
                report += f"  {i}. {pattern.pattern_id}\n"
                report += f"     Score: {pattern.cross_validation_score:.3f}\n"
                report += f"     P-value: {pattern.statistical_significance:.4f}\n"
                report += f"     Effect size: {pattern.effect_size:.3f}\n"
        
        report += f"""
{'='*80}
STATISTICAL SYNTHESIS
{'='*80}
  Overall significance: p = {result.overall_significance:.4f}
  Self-consistency: {result.self_consistency_score:.3f}
  Information preserved: {'Yes' if result.information_preservation > 0.5 else 'No'}

{'='*80}
PATTERN SYNTHESIS
{'='*80}
"""
        
        # Text insights
        text_insights = result.pattern_synthesis['unique_insights']['text_space']
        report += f"  Text space insights:\n"
        for key, value in text_insights.items():
            report += f"    {key}: {value}\n"
        
        # Binary insights
        binary_insights = result.pattern_synthesis['unique_insights']['binary_space']
        report += f"\n  Binary space insights:\n"
        for key, value in binary_insights.items():
            report += f"    {key}: {value}\n"
        
        # Recommendations
        report += f"\n  Recommendations:\n"
        for rec in result.pattern_synthesis['recommendations']:
            report += f"    • {rec}\n"
        
        report += f"""
{'='*80}
OVERALL CONFIDENCE: {result.confidence_score:.1%}
{'='*80}
"""
        
        return report

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_analyze(text: str) -> str:
    """Quick integrated analysis"""
    system = IntegratedPatternSystem()
    result = system.analyze(text, verbose=True)
    return system.generate_report(result)

def analyze_hieroglyphs(hieroglyph_text: str, name: str = "Unknown") -> Dict:
    """Specialized analysis for hieroglyphic text"""
    system = IntegratedPatternSystem()
    result = system.analyze(hieroglyph_text, binary_strategy='adaptive', verbose=True)
    
    return {
        'name': name,
        'hieroglyphs': hieroglyph_text,
        'confidence': result.confidence_score,
        'cross_validated_patterns': len(result.cross_validated),
        'high_confidence_patterns': len(result.high_confidence_patterns),
        'text_patterns': result.text_pattern_count,
        'binary_patterns': result.binary_pattern_count,
        'synthesis': result.pattern_synthesis
    }

# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate():
    """Demonstrate integrated system"""
    print("="*80)
    print("INTEGRATED PATTERN SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Test on Book of the Dead passages
    passages = [
        ('Spell 1 - Entering the Tomb', '𓁹𓈖𓏏𓏤𓀀𓇋𓏲𓂻𓏛𓏥𓀀'),
        ('Spell 15 - Hymn to Ra', '𓇳𓏤𓎟𓋹𓇋𓅱𓏏𓈖𓏤𓀭𓀀'),
        ('Spell 125 - Weighing of Heart', '𓂋𓇋𓎡𓏏𓉐𓊃𓏏𓀀𓁹𓏥')
    ]
    
    system = IntegratedPatternSystem()
    
    for name, hieroglyphs in passages:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {name}")
        print(f"{'='*80}")
        
        result = system.analyze(hieroglyphs, verbose=True)
        
        print(f"\nRESULTS:")
        print(f"  Text patterns: {result.text_pattern_count}")
        print(f"  Binary patterns: {result.binary_pattern_count}")
        print(f"  Cross-validated: {len(result.cross_validated)}")
        print(f"  High confidence: {len(result.high_confidence_patterns)}")
        print(f"  Overall confidence: {result.confidence_score:.1%}")
        
        if result.high_confidence_patterns:
            print(f"\n  Top validated pattern:")
            top = result.high_confidence_patterns[0]
            print(f"    ID: {top.pattern_id}")
            print(f"    Score: {top.cross_validation_score:.3f}")
            print(f"    P-value: {top.statistical_significance:.4f}")

if __name__ == "__main__":
    demonstrate()