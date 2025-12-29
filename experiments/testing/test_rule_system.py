"""
Test script for rule-based threshold system.
Tests parse_rule(), evaluate_rule(), validate_rules(), and get_active_metrics_from_rules().
"""

import numpy as np
import pandas as pd
from auto_inspector import (
    parse_rule,
    evaluate_rule,
    validate_rules,
    get_active_metrics_from_rules,
    DEFAULT_DELETION_RULES
)

def test_parse_rule():
    """Test rule parsing with various formats."""
    print("\n[TEST] parse_rule()")

    test_cases = [
        ('area<1', ('area', '<', 1.0)),
        ('circularity>4', ('circularity', '>', 4.0)),
        ('max_edge>=42', ('max_edge', '>=', 42.0)),
        ('convexity<=42', ('convexity', '<=', 42.0)),
        ('t_rise<0.10', ('t_rise', '<', 0.10)),
        ('  area  <  1  ', ('area', '<', 1.0)),  # Whitespace handling
    ]

    for rule_str, expected in test_cases:
        result = parse_rule(rule_str)
        if result == expected:
            print(f"  PASS: '{rule_str}' -> {result}")
        else:
            print(f"  FAIL: '{rule_str}' -> {result}, expected {expected}")

    # Test invalid rules
    print("\n  Testing invalid rules (should raise ValueError):")
    invalid_cases = [
        'invalid_metric>1',  # Unknown metric
        'area*1',            # Invalid operator
        'area>abc',          # Non-numeric threshold
    ]

    for rule_str in invalid_cases:
        try:
            parse_rule(rule_str)
            print(f"  FAIL: '{rule_str}' should have raised ValueError")
        except ValueError as e:
            print(f"  PASS: '{rule_str}' raised ValueError: {e}")

def test_evaluate_rule():
    """Test rule evaluation with different operators and values."""
    print("\n[TEST] evaluate_rule()")

    # Create test series
    test_data = pd.Series({
        'area': 2.0,
        'circularity': 3.0,
        'max_edge': 50.0,
        'convexity': 40.0,
        't_rise': 0.15,
        'caiman_r_score': 0.10,
        'caiman_snr': 3.5,
        't_off': 2.0
    })

    # Test cases: (metric, operator, threshold, expected_result)
    # Remember: Rules express DELETION, but evaluate_rule returns PASS (True) or FAIL (False)
    # Rule "area<1" means "DELETE if area < 1", so:
    #   - area=2.0 < 1? No -> PASS (True)
    #   - area=0.5 < 1? Yes -> FAIL (False)
    test_cases = [
        ('area', '<', 1.0, True),      # area=2.0 >= 1 -> PASS
        ('area', '<', 3.0, False),     # area=2.0 < 3 -> FAIL (meets deletion condition)
        ('circularity', '>', 4.0, True),   # circularity=3.0 <= 4 -> PASS
        ('circularity', '>', 2.0, False),  # circularity=3.0 > 2 -> FAIL
        ('max_edge', '>=', 60.0, True),    # max_edge=50.0 < 60 -> PASS
        ('max_edge', '>=', 40.0, False),   # max_edge=50.0 >= 40 -> FAIL
        ('convexity', '<=', 30.0, True),   # convexity=40.0 > 30 -> PASS
        ('convexity', '<=', 50.0, False),  # convexity=40.0 <= 50 -> FAIL
    ]

    for metric, op, threshold, expected in test_cases:
        result = evaluate_rule(test_data, metric, op, threshold)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: {metric}{op}{threshold} with value={test_data[metric]} -> {result} (expected {expected})")

    # Test NaN handling
    print("\n  Testing NaN handling:")
    nan_data = pd.Series({
        't_rise': np.nan,
        'caiman_r_score': np.nan,
        'area': np.nan
    })

    nan_cases = [
        ('t_rise', '<', 0.10, False),        # NaN -> FAIL
        ('caiman_r_score', '<', 0.05, False), # NaN -> FAIL
        ('area', '<', 1.0, False),           # NaN -> FAIL (pandas comparison)
    ]

    for metric, op, threshold, expected in nan_cases:
        result = evaluate_rule(nan_data, metric, op, threshold)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: {metric}{op}{threshold} with NaN -> {result} (expected {expected})")

    # Test sentinel values for t_rise and t_off
    print("\n  Testing sentinel values (<0):")
    sentinel_data = pd.Series({
        't_rise': -1.0,
        't_off': -1.0
    })

    sentinel_cases = [
        ('t_rise', '<', 0.10, False),  # -1 (sentinel) -> FAIL
        ('t_off', '<', 1.5, False),    # -1 (sentinel) -> FAIL
    ]

    for metric, op, threshold, expected in sentinel_cases:
        result = evaluate_rule(sentinel_data, metric, op, threshold)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: {metric}{op}{threshold} with value=-1 -> {result} (expected {expected})")

def test_validate_rules():
    """Test rule validation."""
    print("\n[TEST] validate_rules()")

    # Valid rules
    valid_rules = ['area<1', 'circularity>4', 'max_edge>=42']
    try:
        parsed = validate_rules(valid_rules)
        print(f"  PASS: Valid rules parsed: {parsed}")
    except ValueError as e:
        print(f"  FAIL: Valid rules raised error: {e}")

    # Invalid rules
    invalid_rules = ['area<1', 'invalid_metric>1', 'circularity>4']
    try:
        validate_rules(invalid_rules)
        print(f"  FAIL: Invalid rules should have raised ValueError")
    except ValueError as e:
        print(f"  PASS: Invalid rules raised ValueError: {e}")

def test_get_active_metrics():
    """Test extracting active metrics from rules."""
    print("\n[TEST] get_active_metrics_from_rules()")

    rules = ['area<1', 'circularity>4', 'area>=2', 'max_edge>42']
    expected = ['area', 'circularity', 'max_edge']  # Unique, order may vary

    result = get_active_metrics_from_rules(rules)
    result_sorted = sorted(result)
    expected_sorted = sorted(expected)

    if result_sorted == expected_sorted:
        print(f"  PASS: Active metrics: {result}")
    else:
        print(f"  FAIL: Got {result}, expected {expected}")

def test_default_rules():
    """Test that DEFAULT_DELETION_RULES are valid."""
    print("\n[TEST] DEFAULT_DELETION_RULES validation")

    try:
        parsed = validate_rules(DEFAULT_DELETION_RULES)
        print(f"  PASS: Default rules are valid")
        print(f"  Rules: {DEFAULT_DELETION_RULES}")
        print(f"  Parsed: {parsed}")
    except ValueError as e:
        print(f"  FAIL: Default rules are invalid: {e}")

if __name__ == '__main__':
    print("="*60)
    print("RULE-BASED THRESHOLD SYSTEM TESTS")
    print("="*60)

    test_parse_rule()
    test_evaluate_rule()
    test_validate_rules()
    test_get_active_metrics()
    test_default_rules()

    print("\n" + "="*60)
    print("TESTS COMPLETED")
    print("="*60)
