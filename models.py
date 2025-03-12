import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import OpenAI, HuggingFacePipeline
from typing import Dict, Tuple, Any, Optional

class ModelFactory:
    """Factory for creating different types of models"""
    
    def get_model(self, model_id: str, model_type: str, config: Dict) -> Tuple[Any, Any]:
        """Create and return model based on type"""
        if model_type == "hf":
            return self._create_hf_model(model_id, config)
        elif model_type == "openai":
            return self._create_openai_model(model_id, config)
        elif model_type == "custom":
            return self._create_custom_model(model_id, config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _create_hf_model(self, model_id: str, config: Dict) -> Tuple[Any, Any]:
        """Create Hugging Face model"""
        device = 0 if torch.cuda.is_available() else -1
        
        use_8bit = config.get("use_8bit", False)
        use_4bit = config.get("use_4bit", False)
        
        model_kwargs = {}
        if use_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif use_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_quant_type"] = "nf4"
            model_kwargs["device_map"] = "auto"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        gen_pipeline = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            device=device,
            max_length=config.get("max_length", 100),
            temperature=config.get("temperature", 0.1)
        )
        
        llm = HuggingFacePipeline(pipeline=gen_pipeline)
        
        return llm, tokenizer
    
    def _create_openai_model(self, model_id: str, config: Dict) -> Tuple[Any, None]:
        """Create OpenAI model"""
        llm = OpenAI(
            model_name=model_id,
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_length", 100)
        )
        return llm, None
    
    def _create_custom_model(self, model_path: str, config: Dict) -> Tuple[Any, Optional[Any]]:
        """Load custom model (could be local or custom implementation)"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=config.get("max_length", 100),
                temperature=config.get("temperature", 0.1)
            )
            
            llm = HuggingFacePipeline(pipeline=gen_pipeline)
            return llm, tokenizer
        except Exception as e:
            raise NotImplementedError(f"Custom model loading failed: {str(e)}")