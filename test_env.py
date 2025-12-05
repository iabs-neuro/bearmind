"""Quick test of BEARMiND environment imports."""
import warnings
warnings.filterwarnings('ignore')

# All packages required by BEARMiND
REQUIRED_PACKAGES = [
    # Core neuroscience
    ("caiman", "CaImAn calcium imaging"),
    ("driada", "Driada spike detection"),
    # Scientific computing
    ("numpy", "NumPy arrays"),
    ("scipy", "SciPy scientific"),
    ("pandas", "Pandas dataframes"),
    ("h5py", "HDF5 files"),
    # Machine learning
    ("sklearn", "Scikit-learn ML"),
    ("tensorflow", "TensorFlow DL"),
    ("interpret", "InterpretML EBM"),
    ("joblib", "Joblib parallel"),
    # Visualization
    ("bokeh", "Bokeh interactive"),
    ("matplotlib", "Matplotlib plots"),
    ("holoviews", "HoloViews viz"),
    # Image/Video
    ("cv2", "OpenCV image"),
    ("tifffile", "TIFF files"),
    ("moviepy", "MoviePy video"),
    ("imageio", "ImageIO"),
    # Jupyter
    ("ipywidgets", "Jupyter widgets"),
    ("jupyter_bokeh", "Jupyter Bokeh"),
    # Utilities
    ("tqdm", "Progress bars"),
    ("natsort", "Natural sorting"),
    ("sortedcontainers", "Sorted containers"),
    ("pywt", "PyWavelets"),
    ("peakutils", "Peak detection"),
]

def test_imports():
    results = []
    ok_count = 0
    fail_count = 0

    for pkg, desc in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            results.append(f"[OK] {pkg}: {desc}")
            ok_count += 1
        except Exception as e:
            results.append(f"[FAILED] {pkg}: {str(e)[:50]}")
            fail_count += 1

    return results, ok_count, fail_count

if __name__ == "__main__":
    print("BEARMiND Environment Test")
    print("=" * 50)
    results, ok, fail = test_imports()
    for r in results:
        print(f"  {r}")
    print("=" * 50)
    print(f"Results: {ok} OK, {fail} FAILED")
    if fail == 0:
        print("Environment is ready!")
    else:
        print("Some packages need to be installed.")
