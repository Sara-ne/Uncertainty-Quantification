import os
import json
import random
import subprocess

import numpy as np
from tqdm import tqdm
import pandas as pd

from PUNCUncertaintyCalculator import PUNCUncertaintyCalculator
from generateEmbeddings import CLIPEmbeddingExtractor
from kMeansClusterer import KMeansClusterer
from entailment.ImageCaptioner import ImageCaptioner
from entailment.ClusterByEntailment import ClusterByEntailment
from entailment.EntailmentChecker import EntailmentChecker
from entailment.calculateEntropy import SemanticEntropyCalculator
from utils import compute_lpips_diversity
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_JSONL = os.path.join(PROJECT_DIR, "GenEval/prompts/evaluation_metadata.jsonl")
SELECTED_PROMPTS = os.path.join(PROJECT_DIR, "selected_prompts.jsonl")
GEN_IMAGES_DIR = os.path.join(PROJECT_DIR, "generated_images")
EVAL_RESULTS_JSONL = os.path.join(PROJECT_DIR, "generated_eval_results.jsonl")
FINAL_CSV = os.path.join(PROJECT_DIR, "GenEval/geneval_results.csv")
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "GenEval/checkpoints")
MODEL_CONFIG = os.path.join(PROJECT_DIR, "GenEval/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py")

def sample_prompts():
    with open(PROMPT_JSONL) as f:
        lines = [json.loads(l) for l in f.readlines()]
    selected = random.sample(lines, 50)
    with open(SELECTED_PROMPTS, "w") as f:
        for item in selected:
            f.write(json.dumps(item) + "\n")
    print(" Sampled 50 prompts.")

def generate_images():
    cmd = [
        "python", os.path.join(PROJECT_DIR, "GenEval/generation/diffusers_generate.py"),
        SELECTED_PROMPTS,
        "--outdir", GEN_IMAGES_DIR,
        "--n_samples", "15", # 15 images per prompt (50 prompts total)
        "--model", "runwayml/stable-diffusion-v1-5",
        "--batch_size", "3"
    ]
    subprocess.run(cmd, check=True)
    print(" Image generation complete.")

def evaluate_images():
    cmd = [
        "python", os.path.join(PROJECT_DIR, "GenEval/evaluation/evaluate_images.py"),
        GEN_IMAGES_DIR,
        "--outfile", EVAL_RESULTS_JSONL,
        "--model-config", MODEL_CONFIG,
        "--model-path", CHECKPOINT_PATH,
        "--options", "model=mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
    ]
    subprocess.run(cmd, check=True)
    print(" Evaluation complete.")

def load_image(path):
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB")

def compute_uncertainty_metrics():
    with open(SELECTED_PROMPTS) as f:
        prompts = [json.loads(l) for l in f.readlines()]
    eval_df = pd.read_json(EVAL_RESULTS_JSONL, lines=True)

    punc = PUNCUncertaintyCalculator()
    embedder = CLIPEmbeddingExtractor()
    captioner = ImageCaptioner()
    clusterer = KMeansClusterer()
    entropy_calc = SemanticEntropyCalculator()
    entailment_checker = EntailmentChecker()
    entailment_clusterer = ClusterByEntailment(entailment_checker)

    rows = []
    for idx, prompt_data in enumerate(tqdm(prompts)):
        prompt = prompt_data["prompt"]
        prompt_group = eval_df[eval_df['prompt'] == prompt]
        image_paths = sorted(prompt_group["filename"].tolist())
        labels = prompt_group["correct"].tolist()

        images = [load_image(p) for p in image_paths]
        valid = [(p, img) for p, img in zip(image_paths, images) if img is not None]
        if not valid:
            continue
        paths_valid, imgs_valid = zip(*valid)

        captions = [captioner.caption_image(img) for img in imgs_valid]
        punc_result = punc.compute_punc_paper_uncertainty(prompt, captions)
        aleatoric = punc_result["aleatoric uncertainty"]
        epistemic = punc_result["epistemic uncertainty"]
        predictive = aleatoric + epistemic

        similarity = punc.compute_similarity_scores(prompt, captions)
        rouge_l = similarity["avg_rougeL"]
        bertscore_f1 = similarity["avg_bertscore_f1"]

        clip_scores = embedder.get_clip_similarity_scores(prompt, imgs_valid)
        clip_var = float(pd.Series(clip_scores).var())

        lpips = compute_lpips_diversity(imgs_valid)

        embeddings = embedder.get_image_embeddings(imgs_valid)
        cluster_labels, _, _ = clusterer.cluster_images(embeddings)
        sem_entropy = entropy_calc.calculate_entropy(cluster_labels)

        entailment_labels = entailment_clusterer.cluster_images(captions)
        sem_entropy_entailment = entropy_calc.calculate_entropy(entailment_labels)

        accuracy = sum(labels) / len(labels)
        label = 1 if accuracy >= 0.3 else 0

        row = {
            "prompt_id": idx,
            "prompt": prompt,
            "num_images_used": len(imgs_valid),
            "correct_images": sum(labels),
            "total_images": len(labels),
            "accuracy: proportion of correct images for prompt": accuracy,
            "label": label,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "predictive_uncertainty": predictive,
            "semantic_entropy": sem_entropy,
            "semantic_entropy_entailment": sem_entropy_entailment,
            "rouge_l": rouge_l,
            "bertscore_f1": bertscore_f1,
            "clip_score_variance": clip_var,
            "lpips_diversity": lpips
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(FINAL_CSV), exist_ok=True)

    # Save final CSV
    df.to_csv(FINAL_CSV, index=False)
    print(f" Results saved to {FINAL_CSV}")

def run_roc_analysis():
    print(" Starting ROC analysis...")
    cmd = [
        "python", os.path.join(PROJECT_DIR, "roc_analysis.py"),
        "--csv", FINAL_CSV
    ]
    subprocess.run(cmd, check=True)
    print(" ROC analysis complete. Curves saved as PNGs.")


def main():
    sample_prompts()
    generate_images()
    evaluate_images()
    compute_uncertainty_metrics()
    run_roc_analysis()

if __name__ == "__main__":
    main()
