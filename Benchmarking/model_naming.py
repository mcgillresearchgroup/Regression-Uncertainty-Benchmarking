"""Model naming utilities for consistent labeling across the project."""


def get_model_label(wrapper, model):
    """Convert wrapper and model names to clean abbreviated labels.
    
    Args:
        wrapper: Wrapper name (e.g., 'MVE_Ensemble_Averaged')
        model: Model name (e.g., 'MVE_Default')
    
    Returns:
        Clean abbreviated label (e.g., 'MEA-MD')
    """
    wrapper_map = {
        'MVE_Ensemble': 'MEA',  # Legacy name
        'MVE_Ensemble_Averaged': 'MEA',
        'MVE_Ensemble_Multiplicative': 'MEM',
        'MLP_Ensemble': 'MLP',
    }
    model_map = {
        'MVE_Default': 'MD',
        'MVE_Mean_Head_Extension': 'MMH',
        'MLP_Default': 'MLP',
    }
    
    wrapper_label = wrapper_map.get(wrapper, wrapper)
    model_label = model_map.get(model, model)
    
    return f"{wrapper_label}-{model_label}"
