import pickle
from pathlib import Path

# Load one of the files to inspect
test_file = Path('data/raw_compressed/NOF_H01_3D_gsig4_mincorr0.92_minpnr7_estimates.pickle')
print(f'Inspecting: {test_file.name}')
print(f'File size: {test_file.stat().st_size / (1024**2):.1f} MB')
print()

with open(test_file, 'rb') as f:
    est = pickle.load(f)

print(f'Estimates object type: {type(est)}')
print(f'Has A attribute: {hasattr(est, "A")}')
print(f'Has C attribute: {hasattr(est, "C")}')
print(f'Has S attribute: {hasattr(est, "S")}')
print(f'Has YrA attribute: {hasattr(est, "YrA")}')
print()

if hasattr(est, 'A'):
    print(f'A shape: {est.A.shape}')
    print(f'A type: {type(est.A)}')
    print(f'A dtype: {est.A.dtype if hasattr(est.A, "dtype") else "N/A"}')

if hasattr(est, 'C'):
    print(f'C shape: {est.C.shape}')
    print(f'C dtype: {est.C.dtype}')

if hasattr(est, 'S'):
    print(f'S shape: {est.S.shape}')
    print(f'S type: {type(est.S)}')
    if hasattr(est.S, 'dtype'):
        print(f'S dtype: {est.S.dtype}')

if hasattr(est, 'YrA'):
    print(f'YrA shape: {est.YrA.shape}')
    print(f'YrA dtype: {est.YrA.dtype}')

if hasattr(est, 'idx_components_bad'):
    print(f'idx_components_bad: {est.idx_components_bad}')
    print(f'Number of bad components: {len(est.idx_components_bad) if est.idx_components_bad is not None else 0}')

# Check for heavy attributes that should be removed
heavy_attrs = ['AtA', 'AtY_buf', 'CC', 'CY', 'R', 'Yr_buf', 'rho_buf', 'noisyC', 'OASISinstances', 'Ab_dense', 'A_thr']
print('\nHeavy attributes present:')
for attr in heavy_attrs:
    if hasattr(est, attr):
        val = getattr(est, attr)
        if val is not None:
            print(f'  {attr}: present')
