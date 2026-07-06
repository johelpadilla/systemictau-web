import numpy as np
import systemictau as st

def test_iaaft():
    print("Testing IAAFT Surrogates...")
    # Create synthetic data (sine wave + noise)
    t = np.linspace(0, 10, 100)
    X = np.zeros((100, 3))
    X[:, 0] = np.sin(t) + np.random.normal(0, 0.1, 100)
    X[:, 1] = np.sin(t + 0.5) + np.random.normal(0, 0.1, 100)
    X[:, 2] = np.random.normal(0, 1, 100)
    
    # Generate surrogates
    from systemictau.validation import generate_iaaft_surrogates
    surrogates = generate_iaaft_surrogates(X, n_surrogates=5, max_iter=20, seed=42)
    
    assert surrogates.shape == (5, 100, 3)
    
    # Check amplitude distribution preservation
    # If ranked correctly, the sorted values should be identical
    orig_sorted = np.sort(X[:, 0])
    surr_sorted = np.sort(surrogates[0, :, 0])
    np.testing.assert_allclose(orig_sorted, surr_sorted, rtol=1e-5, atol=1e-5)
    print("IAAFT amplitude distribution is preserved! ✅")

def test_surrogate_validation():
    print("Testing Full Validation Pipeline...")
    # Create highly coupled data that drops to noise (to trigger a transition)
    X = np.zeros((150, 4))
    # Epoch 1: Highly correlated (linear trend)
    for i in range(100):
        val = np.random.normal(i*0.1, 0.5)
        X[i, :] = [val, val + np.random.normal(), val*1.2, val*0.8]
        
    # Epoch 2: Complete noise (no correlation)
    for i in range(100, 150):
        X[i, :] = np.random.normal(0, 5, 4)
        
    # Run validation
    res = st.run_surrogate_validation(X, n_surrogates=10, mode="fast", window_size=10, seed=42)
    
    # Generate report
    import systemictau.reporting as reporting
    report = reporting.generate_academic_report(res.real_result, surrogate_result=res)
    print("Report generated successfully. Length:", len(report))
    assert "IAAFT Surrogate Validation" in report
    print("\nPipeline executed successfully! ✅")

if __name__ == "__main__":
    test_iaaft()
    test_surrogate_validation()
