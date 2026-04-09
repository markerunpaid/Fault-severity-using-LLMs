# src/step2_extract_metrics.py
import re, math
import numpy as np
import javalang
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import os

METRIC_COLS = [
    'SLOC', 'CyclomaticComplexity', 'McClureComplexity',
    'NestingDepth', 'ProxyIndentation', 'Readability',
    'FanOut', 'HalsteadDifficulty', 'HalsteadEffort',
    'MaintainabilityIndex'
]

# ── Individual metric functions ───────────────────────────────────────────────

def compute_sloc(code):
    lines = code.splitlines()
    in_block = False
    count = 0
    for line in lines:
        s = line.strip()
        if in_block:
            if '*/' in s: in_block = False
            continue
        if s.startswith('/*') or s.startswith('/**'):
            in_block = True; continue
        if s.startswith('//') or s == '':
            continue
        count += 1
    return max(count, 1)

BRANCH_KW = re.compile(r'\b(if|else\s+if|for|while|case|catch|&&|\|\||\?)\b')
def compute_cyclomatic(code):
    return 1 + len(BRANCH_KW.findall(code))

LOGICAL_OPS = re.compile(r'(&&|\|\|)')
def compute_mcclure(code):
    return len(LOGICAL_OPS.findall(code))

def compute_nesting(code):
    max_d = d = 0
    for c in code:
        if c == '{':
            d += 1
            max_d = max(max_d, d)
        elif c == '}':
            d = max(d - 1, 0)
    return max_d

def compute_pi(code):
    levels = [len(l) - len(l.lstrip()) for l in code.splitlines() if l.strip()]
    return float(np.std(levels)) if len(levels) > 1 else 0.0

def compute_readability(code):
    lines = [l for l in code.splitlines() if l.strip()]
    return float(np.mean([len(l) for l in lines])) if lines else 0.0

CTRL_KW = {'if','for','while','switch','catch','return',
           'assert','throw','new','super','this'}
CALL_RE = re.compile(r'\b([a-zA-Z_]\w*)\s*\(')
def compute_fanout(code):
    return sum(1 for c in CALL_RE.findall(code) if c.lower() not in CTRL_KW)

def _tokenize_java(code):
    try:
        toks = list(javalang.tokenizer.tokenize(code))
        ops  = [t.value for t in toks
                if type(t).__name__ in ('Operator', 'Separator')]
        opds = [t.value for t in toks
                if type(t).__name__ in ('Integer', 'FloatingPoint', 'String',
                                        'Boolean', 'Null', 'Identifier',
                                        'BasicType', 'Keyword')]
        return ops, opds
    except Exception:
        ops  = re.findall(r'[+\-*/%=!<>&|^~?:]+', code)
        opds = re.findall(r'\b\w+\b', code)
        return ops, opds

def compute_halstead(code):
    ops, opds = _tokenize_java(code)
    n1, n2 = len(set(ops)), len(set(opds))
    N1, N2 = len(ops),     len(opds)
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0
    vocab  = n1 + n2
    length = N1 + N2
    diff   = (n1 / 2) * (N2 / n2)
    vol    = length * math.log2(vocab) if vocab > 1 else 0.0
    effort = diff * vol
    return round(diff, 4), round(effort, 4)

def compute_mi(code):
    sloc       = compute_sloc(code)
    cc         = compute_cyclomatic(code)
    ops, opds  = _tokenize_java(code)
    n1, n2     = len(set(ops)), len(set(opds))
    N1, N2     = len(ops),     len(opds)
    vocab      = n1 + n2
    length     = N1 + N2
    vol        = length * math.log2(vocab) if vocab > 1 else 1.0
    mi         = (171
                  - 5.2  * math.log(max(vol,  1))
                  - 0.23 * cc
                  - 16.2 * math.log(max(sloc, 1)))
    return round(max(0.0, min(100.0, mi)), 4)

# ── Master extractor for one method ──────────────────────────────────────────

def extract_metrics(code):
    code = str(code)
    diff, effort = compute_halstead(code)
    return {
        'SLOC':                  compute_sloc(code),
        'CyclomaticComplexity':  compute_cyclomatic(code),
        'McClureComplexity':     compute_mcclure(code),
        'NestingDepth':          compute_nesting(code),
        'ProxyIndentation':      compute_pi(code),
        'Readability':           compute_readability(code),
        'FanOut':                compute_fanout(code),
        'HalsteadDifficulty':    diff,
        'HalsteadEffort':        effort,
        'MaintainabilityIndex':  compute_mi(code),
    }

# ── Batch extraction ──────────────────────────────────────────────────────────

def extract_all(df, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_metrics)(code)
        for code in tqdm(df['method_code'], desc="Extracting metrics")
    )
    metrics_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), metrics_df], axis=1)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    for split in ['train', 'test']:
        print(f"\n--- {split} ---")
        df  = pd.read_csv(f"data/{split}_raw.csv")
        df  = extract_all(df, n_jobs=-1)
        out = f"data/{split}_with_metrics.csv"
        df.to_csv(out, index=False)
        print(f"Saved → {out}  shape: {df.shape}")
        print(df[METRIC_COLS].describe().round(2))