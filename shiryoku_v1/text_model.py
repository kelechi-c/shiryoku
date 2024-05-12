from transformers import SiglipTextModel, SiglipTokenizer, SiglipTextConfig

text_config = SiglipTextConfig()
text_model = SiglipTextModel(text_config)
