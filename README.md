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

1. **Clone this repository** and install dependencies:
git clone https://github.com/Sara-ne/Uncertainty-Quantification.git
cd Uncertainty-Quantification

2. **Clone and install GenEval** (required for correctness evaluation):
git clone https://github.com/djghosh13/geneval/tree/main
cd geneval
pip install -e .

Make sure the `geneval/` folder is either in this project directory or added to your Python path.

3. **Download models**:
- Stable Diffusion v1.5
- BLIP (image captioning)
- CLIP via openai/clip



Output Overview

For each prompt, the pipeline produces:
- Correctness score via GenEval
- Semantic entropy (based on CLIP embeddings and BLIP captions)
- Aleatoric and epistemic uncertainty via PUNC
- Additional metrics:
  - CLIP Score Variance
  - LPIPS Diversity
  - BERTScore
  - ROUGE


Citation

If you use this codebase, please cite the following:
- Ghosh et al., *GENEVAL: An Object-Focused Framework for Evaluating Text-to-Image Alignment* (2023)
- Farquhar et al., *Detecting hallucinations in large language models using semantic entropy* (2024)
- Franchi et al., *Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation* (2024)
