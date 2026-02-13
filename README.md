Uncertainty Evaluation for Text-to-Image Generation

This project investigates how to quantify uncertainty in text-to-image (T2I) models like Stable Diffusion. It combines image-based and text-based semantic entropy with predictive uncertainty decomposition using the [GenEval](https://github.com/djghosh13/geneval) framework.

Key Contributions

- Adapted the **semantic entropy** method (originally for text) to evaluate distributional uncertainty in T2I outputs.
- Used **GenEval** to label generated images as correct/incorrect based on prompt-object alignment (type, count, position, and attributes).
- Integrated the **PUNC framework** to decompose predictive uncertainty into:
  - **Aleatoric Uncertainty** (prompt ambiguity)
  - **Epistemic Uncertainty** (model knowledge gaps)
- Performed quantitative and qualitative analyses of uncertainty metrics on a structured prompt dataset.

Setup Instructions

**Requirements**
Python 3.9 or 3.10 recommended
CUDA-enabled GPU recommended (tested with CUDA 11.3)

1. **Clone this repository**:
- git clone https://github.com/Sara-ne/Uncertainty-Quantification.git
- cd Uncertainty-Quantification

2. **Create virtual environment**:
python -m venv venv
source venv/bin/activate

3. **Install dependencies**:
- pip install -r requirements.txt

4. **Clone and install GenEval** (required for correctness evaluation):
- git clone https://github.com/djghosh13/geneval.git GenEval
- cd GenEval
- pip install -e .

Make sure the `GenEval/` folder is either in this project directory or added to your Python path.

5. **Download GenEval evaluation checkpoints**:
The evaluation stage requires Mask2Former checkpoints used by GenEval.
Follow the instructions in:
https://github.com/djghosh13/geneval
Place checkpoints in: GenEval/checkpoints/

6. **Running the pipeline**:
Run the full pipeline:
python run_pipeline.py

This will:
- sample 50 prompts
- generate 15 images per prompt using Stable Diffusion v1.5
- evaluate images with GenEval
- compute uncertainty metrics
- save results and ROC curves



Output Overview

The pipeline produces:
- Generated images → generated_images/
- Evaluation results → generated_eval_results.jsonl
- Final metrics → GenEval/geneval_results.csv
- ROC curves → PNG files
Metrics include:
- GenEval correctness score
- Semantic entropy (CLIP embeddings + captions)
- Aleatoric and epistemic uncertainty
- CLIP score variance
- LPIPS diversity
- BERTScore
- ROUGE


Citation

- Ghosh et al., *GENEVAL: An Object-Focused Framework for Evaluating Text-to-Image Alignment* (2023)
- Farquhar et al., *Detecting hallucinations in large language models using semantic entropy* (2024)
- Franchi et al., *Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation* (2024)
