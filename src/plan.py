"""Predeclared condition matrix. No test-driven additions."""
def conditions():
    rows = []
    for fraction in [0.01, 0.05, 0.1, 0.25, 1.0]:
        for seed in ([42] if fraction == 1 else [42, 52, 62]):
            for model in ['scratch', 'pretrained']:
                rows.append(dict(model=model, fraction=fraction, seed=seed))
    for fraction in [0.05, 0.1, 1.0]:
        for seed in ([42] if fraction == 1 else [42, 52, 62]):
            rows.append(dict(model='linear_probe', fraction=fraction, seed=seed))
    return rows


def run_name(row):
    return f'{row["model"]}_f{row["fraction"]:g}_s{row["seed"]}'
