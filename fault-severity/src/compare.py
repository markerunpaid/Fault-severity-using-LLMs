import json, os, pandas as pd

rows = []
for f in sorted(os.listdir('results')):
    if f.endswith('_results.json'):
        r = json.load(open(f'results/{f}'))
        rows.append({
            'Model':       r['model'].upper(),
            'Accuracy':    r['Accuracy'],
            'Macro F1':    r['Macro F1'],
            'Weighted F1': r['Weighted F1'],
            'MCC':         r['MCC'],
            'G-Mean':      r['G-Mean'],
        })

df = pd.DataFrame(rows).sort_values('Macro F1', ascending=False).reset_index(drop=True)
df.index += 1

print()
print('='*72)
print('FINAL MODEL COMPARISON — Fault Severity Prediction')
print('='*72)
print(df.to_string())
print('='*72)
print(f'Best overall : {df.iloc[0]["Model"]}')
print(f'Macro F1     : {df.iloc[0]["Macro F1"]}')
print(f'Improvement over CodeBERT (baseline): +{round(df.iloc[0]["Macro F1"] - df[df["Model"]=="CODEBERT"]["Macro F1"].values[0], 4)}')
