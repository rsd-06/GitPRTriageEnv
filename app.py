import os
import json
import gradio as gr
import pandas as pd
import requests
import re

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
HF_TOKEN = os.getenv("HF_TOKEN")

model = None
tokenizer = None

def load_metrics():
    try:
        baseline_path = "evaluation/pre_training/baseline_summary.json"
        trained_path = "evaluation/post_training/trained_summary.json"
        
        baseline_data = {}
        if os.path.exists(baseline_path):
            with open(baseline_path, "r") as f:
                baseline_data = json.load(f)
                
        trained_data = {}
        if os.path.exists(trained_path):
            with open(trained_path, "r") as f:
                trained_data = json.load(f)
                
        difficulties = ["easy", "medium", "hard"]
        records = []
        for diff in difficulties:
            b_mean = baseline_data.get(diff, {}).get("mean_reward", 0.0)
            t_mean = trained_data.get(diff, {}).get("mean_reward", 0.0)
            records.append({
                "Difficulty": diff.capitalize(),
                "Baseline Mean": round(b_mean, 4),
                "Trained Mean": round(t_mean, 4),
                "Delta": round(t_mean - b_mean, 4)
            })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame([{"Difficulty": "Error loading metrics"}])

def run_inference(pr_title, pr_description, proposed_code, context_snippet, task_level):
    global model, tokenizer
    try:
        if model is None:
            try:
                from unsloth import FastLanguageModel
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name="evaluation/checkpoints/final_adapter/",
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=True,
                )
                FastLanguageModel.for_inference(model)
            except Exception as e:
                return "0.0", json.dumps({"error": f"Failed to load model: {str(e)}"}), "{}"

        prompt = f"""You are a PR reviewer. Review the following PR:
Title: {pr_title}
Description: {pr_description}
Proposed Code: {proposed_code}
Context: {context_snippet}

Please output your evaluation in JSON format containing 'decision' and 'reasoning' keys.
"""
        try:
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
            response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        except Exception as e:
            return "0.0", json.dumps({"error": f"Inference failed: {str(e)}"}), "{}"
        
        action_json = "{}"
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                action_json = match.group(0)
            action = json.loads(action_json)
        except Exception:
            action = {"raw_output": response_text}
            action_json = response_text

        try:
            reset_resp = requests.post(f"{ENV_URL}/reset", json={"level": task_level.lower()})
            reset_resp.raise_for_status()
            
            step_resp = requests.post(f"{ENV_URL}/step", json={"action": action})
            step_resp.raise_for_status()
            
            result = step_resp.json()
            reward = result.get("reward", 0.0)
            info = result.get("info", {})
            
        except Exception as e:
            reward = 0.0
            info = {"error": f"Environment interaction failed: {str(e)}"}

        return str(reward), json.dumps(info, indent=2), action_json

    except Exception as e:
        return "0.0", json.dumps({"error": str(e)}), "{}"

def create_app():
    with gr.Blocks(title="GitPRTriage Results & Evaluation") as app:
        gr.Markdown("# GitPRTriage Environment")
        
        with gr.Tab("Results Dashboard"):
            gr.Markdown("## Performance Comparison")
            with gr.Row():
                try:
                    comp_img = "evaluation/plots/comparison.png"
                    gr.Image(comp_img if os.path.exists(comp_img) else None, label="Baseline vs Trained Rewards")
                except Exception:
                    gr.Markdown("*Comparison plot not available.*")
            
            gr.Markdown("## Metrics")
            metrics_table = gr.Dataframe(value=load_metrics())
            
            gr.Markdown("## Training Logs")
            with gr.Row():
                try:
                    train_img = "evaluation/plots/training_curve.png"
                    gr.Image(train_img if os.path.exists(train_img) else None, label="Training Curve")
                except Exception:
                    pass
                try:
                    loss_img = "evaluation/plots/loss_curve.png"
                    gr.Image(loss_img if os.path.exists(loss_img) else None, label="Loss Curve")
                except Exception:
                    pass

        with gr.Tab("Live Evaluation"):
            with gr.Row():
                with gr.Column():
                    pr_title = gr.Textbox(label="PR Title")
                    pr_description = gr.Textbox(label="PR Description", lines=3)
                    proposed_code = gr.Textbox(label="Proposed Code", lines=5)
                    context_snippet = gr.Textbox(label="Context Snippet", lines=5)
                    task_level = gr.Dropdown(choices=["Easy", "Medium", "Hard"], value="Easy", label="Task Level")
                    submit_btn = gr.Button("Evaluate")
                
                with gr.Column():
                    reward_score = gr.Textbox(label="Reward Score")
                    reward_breakdown = gr.Textbox(label="Reward Breakdown Dict", lines=5)
                    raw_output = gr.Textbox(label="Model Raw JSON Output", lines=10)
            
            submit_btn.click(
                fn=run_inference,
                inputs=[pr_title, pr_description, proposed_code, context_snippet, task_level],
                outputs=[reward_score, reward_breakdown, raw_output]
            )
            
    return app

if __name__ == "__main__":
    try:
        app = create_app()
        app.launch()
    except Exception as e:
        print(f"Failed to launch app: {e}")
