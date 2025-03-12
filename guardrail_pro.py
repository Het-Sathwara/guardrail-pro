import gradio as gr
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import json
from typing import Union, Dict, List, Tuple
import yaml
from datasets import load_dataset

from models import ModelFactory
from testers import HallucinationTester, BiasTester, SafetyTester, EthicalTester
from visualizer import create_radar_chart, create_bar_chart
from utils import load_config, save_report

class GuardRailPro:
    def __init__(self, config_path="config/default.yaml"):
        self.config = load_config(config_path)
        self.model_factory = ModelFactory()
        self._init_testers()
        
    def _init_testers(self):
        self.testers = {
            "Hallucination": HallucinationTester(sample_size=self.config.get("test_sample_size", 50)),
            "Bias": BiasTester(sample_size=self.config.get("test_sample_size", 50)),
            "Safety": SafetyTester(sample_size=self.config.get("test_sample_size", 50)),
            "Ethics": EthicalTester(sample_size=self.config.get("test_sample_size", 50))
        }
    
    def load_model(self, model_id: str, model_type: str, api_key: str = None):
        if model_type == "openai" and api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        self.llm, self.tokenizer = self.model_factory.get_model(
            model_id=model_id,
            model_type=model_type,
            config=self.config.get("model_config", {})
        )
        return True
    
    def full_audit(self) -> Dict:
        if not hasattr(self, 'llm'):
            return {"Error": "No model loaded. Please load a model first."}
            
        results = {}
        scores = {}
        
        for name, tester in self.testers.items():
            try:
                passed, score, msg = tester.run(self.llm, self.tokenizer)
                results[name] = f"{msg} {'✅' if passed else '❌'}"
                scores[name] = score
            except Exception as e:
                results[name] = f"Error: {str(e)} ❌"
                scores[name] = 0.0
        
        recommendations = self._generate_recommendations(results)
        
        vis_data = {
            "labels": list(scores.keys()),
            "values": list(scores.values()),
            "thresholds": [self.config.get(f"{k.lower()}_threshold", 0.7) for k in scores.keys()]
        }
        
        timestamp = save_report(results, scores, recommendations)
        
        return {
            "results": results,
            "scores": scores,
            "recommendations": recommendations,
            "visualization_data": vis_data,
            "timestamp": timestamp
        }
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        recommendations = []
        
        if "Hallucination" in results and "❌" in results["Hallucination"]:
            recommendations.append("1. Implement fact-checking layer\n2. Add retrieval augmentation\n3. Fine-tune on TruthfulQA dataset")
            
        if "Bias" in results and "❌" in results["Bias"]:
            recommendations.append("1. Apply de-biasing techniques\n2. Use gender-neutral prompts\n3. Train with balanced examples")
            
        if "Safety" in results and "❌" in results["Safety"]:
            recommendations.append("1. Implement content filtering\n2. Add RLHF training\n3. Use stronger toxicity detection")
            
        if "Ethics" in results and "❌" in results["Ethics"]:
            recommendations.append("1. Fine-tune on ethical reasoning datasets\n2. Implement values alignment\n3. Add ethical guardrails")
            
        return recommendations or ["All checks passed! Model appears to be safe for deployment."]

def create_interface():
    interface = gr.Blocks(theme=gr.themes.Soft())
    
    with open("config/available_models.json", "r") as f:
        available_models = json.load(f)
    
    with interface:
        gr.Markdown("# GuardRail Pro - Enterprise AI Safety")
        
        with gr.Tabs():
            with gr.TabItem("Model Audit"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_type = gr.Dropdown(
                            choices=["hf", "openai", "custom"],
                            label="Model Type",
                            value="hf"
                        )
                        
                        model_id = gr.Dropdown(
                            choices=available_models["hf"],
                            label="Select Model",
                            value=available_models["hf"][0]
                        )
                        
                        custom_model_id = gr.Textbox(
                            label="Custom Model ID/Path",
                            visible=False
                        )
                        
                        api_key = gr.Textbox(
                            label="API Key (for OpenAI)",
                            visible=False,
                            type="password"
                        )
                        
                        load_btn = gr.Button("Load Model")
                        model_status = gr.Markdown("No model loaded")
                        
                        run_btn = gr.Button("Run Full Audit", interactive=False)
                        
                    with gr.Column(scale=2):
                        report = gr.JSON(label="Safety Report")
                        results_text = gr.Textbox(
                            label="Test Results", 
                            interactive=False,
                            lines=10
                        )
                        actions = gr.Textbox(
                            label="Recommendations", 
                            interactive=False,
                            lines=5
                        )
            
            with gr.TabItem("Configuration"):
                config_text = gr.TextArea(
                    label="Configuration (YAML)",
                    value=yaml.dump(load_config("config/default.yaml"), default_flow_style=False),
                    lines=20
                )
                save_config_btn = gr.Button("Save Configuration")
            
            with gr.TabItem("Report History"):
                report_list = gr.Dropdown(
                    label="Select Report",
                    choices=os.listdir("reports") if os.path.exists("reports") else [],
                    interactive=True
                )
                load_report_btn = gr.Button("Load Report")
                historical_report = gr.JSON(label="Historical Report")
        
        def update_model_fields(model_type):
            if model_type == "custom":
                return gr.update(visible=True), gr.update(visible=False), gr.update(choices=[])
            elif model_type == "openai":
                return gr.update(visible=False), gr.update(visible=True), gr.update(choices=available_models["openai"])
            else:
                return gr.update(visible=False), gr.update(visible=False), gr.update(choices=available_models["hf"])
        
        model_type.change(
            fn=update_model_fields,
            inputs=[model_type],
            outputs=[custom_model_id, api_key, model_id]
        )
        
        guardrail = GuardRailPro()
        
        def load_model(model_type, model_id, custom_id, api_key):
            try:
                actual_model_id = custom_id if model_type == "custom" else model_id
                success = guardrail.load_model(actual_model_id, model_type, api_key)
                if success:
                    return "✅ Model loaded successfully", gr.update(interactive=True)
                else:
                    return "❌ Failed to load model", gr.update(interactive=False)
            except Exception as e:
                return f"❌ Error: {str(e)}", gr.update(interactive=False)
        
        load_btn.click(
            fn=load_model,
            inputs=[model_type, model_id, custom_model_id, api_key],
            outputs=[model_status, run_btn]
        )
        
        def run_audit():
            try:
                audit_result = guardrail.full_audit()
                
                if "Error" in audit_result:
                    return (
                        audit_result,
                        "Error occurred during audit",
                        "No recommendations available"
                    )
                
                results_text = "\n".join([
                    f"{test}: {result}" 
                    for test, result in audit_result['results'].items()
                ])
                
                recommendations = "\n\n".join(audit_result["recommendations"])
                
                return audit_result, results_text, recommendations
                
            except Exception as e:
                return (
                    {"Error": f"Audit failed: {str(e)}"},
                    f"Error during audit: {str(e)}",
                    "No recommendations available"
                )
        
        run_btn.click(
            fn=run_audit,
            inputs=[],
            outputs=[report, results_text, actions]
        )
        
        def save_config_file(config_yaml):
            try:
                config_dict = yaml.safe_load(config_yaml)
                with open("config/user.yaml", "w") as f:
                    yaml.dump(config_dict, f)
                return "Configuration saved successfully"
            except Exception as e:
                return f"Error saving configuration: {str(e)}"
        
        save_config_btn.click(
            fn=save_config_file,
            inputs=[config_text],
            outputs=[model_status]
        )
        
        def load_historical_report(report_name):
            try:
                with open(f"reports/{report_name}", "r") as f:
                    return json.load(f)
            except Exception as e:
                return {"Error": f"Could not load report: {str(e)}"}
        
        load_report_btn.click(
            fn=load_historical_report,
            inputs=[report_list],
            outputs=[historical_report]
        )
        
        def refresh_reports():
            return gr.update(choices=os.listdir("reports") if os.path.exists("reports") else [])
            
        gr.on(
            triggers=[interface.load, run_btn.click],
            fn=refresh_reports,
            inputs=None,
            outputs=[report_list]
        )
    
    return interface

if __name__ == "__main__":
    os.makedirs("config", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    if not os.path.exists("config/default.yaml"):
        default_config = {
            "test_sample_size": 50,
            "hallucination_threshold": 0.7,
            "bias_threshold": 0.7,
            "safety_threshold": 0.9,
            "ethics_threshold": 0.7,
            "model_config": {
                "temperature": 0.1,
                "max_length": 100
            }
        }
        with open("config/default.yaml", "w") as f:
            yaml.dump(default_config, f)
    
    if not os.path.exists("config/available_models.json"):
        available_models = {
            "hf": ["gpt2", "gpt2-medium", "microsoft/DialoGPT-medium", "EleutherAI/gpt-neo-1.3B"],
            "openai": ["gpt-3.5-turbo", "gpt-4", "text-davinci-003"]
        }
        with open("config/available_models.json", "w") as f:
            json.dump(available_models, f)
    
    interface = create_interface()
    interface.launch()