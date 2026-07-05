import numpy as np

def evaluate_early_warning(predictions: np.ndarray, truth: np.ndarray, threshold: float = 0.5):
    """
    Evaluates early warning signals (like Systemic Tau or Ontological Ascent flags)
    against ground truth outbreak events.
    
    Parameters:
    -----------
    predictions : np.ndarray
        1D array of warning signals (probabilities or binary flags).
    truth : np.ndarray
        1D array of ground truth (1 for outbreak, 0 for normal).
    threshold : float
        Threshold for binary classification if predictions are continuous.
        
    Returns:
    --------
    dict
        Metrics including 'lead_time', 'false_alarm_rate', 'precision', 'recall', 'auc'.
    """
    try:
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
    except ImportError:
        raise ImportError("Validation module requires scikit-learn. Run 'pip install systemictau[validation]'")
        
    preds_bin = (predictions >= threshold).astype(int)
    
    # Calculate basic metrics
    auc = roc_auc_score(truth, predictions) if len(np.unique(truth)) > 1 else np.nan
    precision = precision_score(truth, preds_bin, zero_division=0)
    recall = recall_score(truth, preds_bin, zero_division=0)
    
    cm = confusion_matrix(truth, preds_bin)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        far = np.nan
        
    # Calculate lead time (simplified: average distance from warning to next outbreak)
    # This is a naive implementation; proper lead time requires time series matching.
    lead_times = []
    outbreak_idx = np.where(truth == 1)[0]
    warning_idx = np.where(preds_bin == 1)[0]
    
    for w_idx in warning_idx:
        future_outbreaks = outbreak_idx[outbreak_idx > w_idx]
        if len(future_outbreaks) > 0:
            lead_times.append(future_outbreaks[0] - w_idx)
            
    avg_lead_time = np.mean(lead_times) if lead_times else 0.0
    
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'false_alarm_rate': far,
        'avg_lead_time_steps': avg_lead_time
    }
