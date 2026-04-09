"""
Extracts 10 method-level software metrics:
  SLOC, CyclomaticComplexity, McClureComplexity, NestingDepth,
  ProxyIndentation, Readability, FanOut,
  HalsteadDifficulty, HalsteadEffort, MaintainabilityIndex
"""

import re
import math
import numpy as np
import javalang
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. SLOC — Source Lines of Code
# ─────────────────────────────────────────────────────────────────────────────
def compute_sloc(code: str) -> int:
    """Count non-blank, non-comment lines."""
    lines = code.splitlines()
    in_block = False
    count = 0
    for line in lines:
        stripped = line.strip()
        if in_block:
            if '*/' in stripped:
                in_block = False
            continue
        if stripped.startswith('/*') or stripped.startswith('/**'):
            in_block = True
            continue
        if stripped.startswith('//') or stripped == '':
            continue
        count += 1
    return max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cyclomatic Complexity (approximated via keyword counts)
# ─────────────────────────────────────────────────────────────────────────────
BRANCH_KEYWORDS = re.compile(
    r'\b(if|else\s+if|for|while|case|catch|&&|\|\||\?)\b'
)

def compute_cyclomatic(code: str) -> int:
    """McCabe complexity: 1 + number of decision points."""
    return 1 + len(BRANCH_KEYWORDS.findall(code))


# ─────────────────────────────────────────────────────────────────────────────
# 3. McClure Complexity — logical operators in conditions
# ─────────────────────────────────────────────────────────────────────────────
LOGICAL_OPS = re.compile(r'(&&|\|\|)')

def compute_mcclure(code: str) -> int:
    return len(LOGICAL_OPS.findall(code))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Maximum Nesting Depth (brace tracking)
# ─────────────────────────────────────────────────────────────────────────────
def compute_nesting_depth(code: str) -> int:
    max_depth = 0
    depth = 0
    for char in code:
        if char == '{':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == '}':
            depth = max(depth - 1, 0)
    return max_depth


# ─────────────────────────────────────────────────────────────────────────────
# 5. Proxy Indentation — std dev of indentation levels
# ─────────────────────────────────────────────────────────────────────────────
def compute_proxy_indentation(code: str) -> float:
    levels = []
    for line in code.splitlines():
        if line.strip() == '':
            continue
        indent = len(line) - len(line.lstrip())
        levels.append(indent)
    return float(np.std(levels)) if len(levels) > 1 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Readability — average line length (proxy)
# ─────────────────────────────────────────────────────────────────────────────
def compute_readability(code: str) -> float:
    lines = [l for l in code.splitlines() if l.strip()]
    if not lines:
        return 0.0
    return float(np.mean([len(l) for l in lines]))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Fan-Out — method calls (excluding control-flow keywords)
# ─────────────────────────────────────────────────────────────────────────────
CONTROL_KEYWORDS = {'if', 'for', 'while', 'switch', 'catch', 'return',
                    'assert', 'throw', 'new', 'super', 'this'}
METHOD_CALL = re.compile(r'\b([a-zA-Z_]\w*)\s*\(')

def compute_fan_out(code: str) -> int:
    calls = METHOD_CALL.findall(code)
    return sum(1 for c in calls if c.lower() not in CONTROL_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# 8 & 9. Halstead Difficulty & Effort (using javalang for lexical precision)
# ─────────────────────────────────────────────────────────────────────────────
JAVA_OPERATORS = {
    '+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=',
    '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '>>>', '++', '--',
    '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '>>>=',
    '?', ':', '->', '::', '.'
}

def _tokenize_java(code: str):
    """Use javalang tokenizer; fallback to regex on parse error."""
    try:
        tokens = list(javalang.tokenizer.tokenize(code))
        operators, operands = [], []
        for tok in tokens:
            t = type(tok).__name__
            if t in ('Operator', 'Separator'):
                operators.append(tok.value)
            elif t in ('Integer', 'FloatingPoint', 'String',
                       'Boolean', 'Null', 'Identifier', 'BasicType',
                       'Keyword'):
                operands.append(tok.value)
        return operators, operands
    except Exception:
        # Fallback: rough split
        ops = re.findall(r'[+\-*/%=!<>&|^~?:]+', code)
        operands = re.findall(r'\b\w+\b', code)
        return ops, operands

def compute_halstead(code: str):
    operators, operands = _tokenize_java(code)
    n1 = len(set(operators))          # distinct operators
    n2 = len(set(operands))           # distinct operands
    N1 = len(operators)               # total operators
    N2 = len(operands)                # total operands

    if n2 == 0 or n1 == 0:
        return 0.0, 0.0

    vocabulary = n1 + n2
    length = N1 + N2
    difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0.0
    volume = length * math.log2(vocabulary) if vocabulary > 1 else 0.0
    effort = difficulty * volume
    return round(difficulty, 4), round(effort, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Maintainability Index
# ─────────────────────────────────────────────────────────────────────────────
def compute_maintainability_index(code: str) -> float:
    """
    MI = 171 - 5.2*ln(HalsteadVolume) - 0.23*CyclomaticComplexity - 16.2*ln(SLOC)
    Clamped to [0, 100].
    """
    sloc = compute_sloc(code)
    cc   = compute_cyclomatic(code)
    _, effort = compute_halstead(code)

    operators, operands = _tokenize_java(code)
    n1 = len(set(operators)); n2 = len(set(operands))
    N1 = len(operators);      N2 = len(operands)
    vocab = n1 + n2; length = N1 + N2
    volume = length * math.log2(vocab) if vocab > 1 else 1.0

    mi = 171 - 5.2 * math.log(max(volume, 1)) \
             - 0.23 * cc \
             - 16.2 * math.log(max(sloc, 1))
    return round(max(0.0, min(100.0, mi)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Master extractor
# ─────────────────────────────────────────────────────────────────────────────
METRIC_NAMES = [
    'SLOC', 'CyclomaticComplexity', 'McClureComplexity',
    'NestingDepth', 'ProxyIndentation', 'Readability',
    'FanOut', 'HalsteadDifficulty', 'HalsteadEffort',
    'MaintainabilityIndex'
]

def extract_metrics(code: str) -> dict:
    diff, effort = compute_halstead(code)
    return {
        'SLOC':                  compute_sloc(code),
        'CyclomaticComplexity':  compute_cyclomatic(code),
        'McClureComplexity':     compute_mcclure(code),
        'NestingDepth':          compute_nesting_depth(code),
        'ProxyIndentation':      compute_proxy_indentation(code),
        'Readability':           compute_readability(code),
        'FanOut':                compute_fan_out(code),
        'HalsteadDifficulty':    diff,
        'HalsteadEffort':        effort,
        'MaintainabilityIndex':  compute_maintainability_index(code),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch extraction (parallelized)
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

def extract_all_metrics(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_metrics)(code)
        for code in tqdm(df['method_code'], desc="Extracting metrics")
    )
    metrics_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), metrics_df], axis=1)

if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    df = extract_all_metrics(df)
    df.to_csv("data/train_with_metrics.csv", index=False)
    print("Done. Sample:")
    print(df[METRIC_NAMES].describe())