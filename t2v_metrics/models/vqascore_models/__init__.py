from .gpt4v_model import GPT4V_MODELS, GPT4VModel
from ...constants import HF_CACHE_DIR

ALL_VQA_MODELS = [
    GPT4V_MODELS,
]

def list_all_vqascore_models():
    return [model for models in ALL_VQA_MODELS for model in models]

def get_vqascore_model(model_name, device='cpu', cache_dir=HF_CACHE_DIR, **kwargs):
    assert model_name in list_all_vqascore_models()
    if model_name in GPT4V_MODELS:
        return GPT4VModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()