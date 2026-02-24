from rouge_score import rouge_scorer
import bert_score
import spacy

nlp = spacy.load('en_core_web_sm')


class PUNCUncertaintyCalculator:
    """
    Implements the PUNC framework to quantify uncertainty in text-to-image generation.
    """

    def __init__(self, num_samples=10, epsilon=1e-6):
        self.num_samples = num_samples
        self.epsilon = epsilon

    @staticmethod
    def extract_concepts(text):
        """
        extract noun-based concepts from a sentence using lemmatization.
        """
        doc = nlp(text.lower())
        return set(token.lemma_ for token in doc if token.pos_ == 'NOUN')

    def compute_punc_paper_uncertainty(self, prompt, captions):
        prompt_concepts = self.extract_concepts(prompt)
        precisions = []
        recalls = []

        for caption in captions:
            caption_concepts = self.extract_concepts(caption)

            intersection = prompt_concepts & caption_concepts

            # aleatoric uncertainty: precision = intersection / caption concepts
            if caption_concepts:
                precision = len(intersection) / len(caption_concepts)
            else:
                precision = 0.0

            # epistemic uncertainty = recall = intersection / prompt concepts
            if prompt_concepts:
                recall = len(intersection) / len(prompt_concepts)
            else:
                recall = 0.0

            precisions.append(precision)
            recalls.append(recall)

        # compute average
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)

        aleatoric = 1 - avg_precision
        epistemic = 1 - avg_recall

        return {
            "aleatoric uncertainty": aleatoric,
            "epistemic uncertainty": epistemic
        }

    # using ROUGE and BERTScore to evaluate how well generated image captions match the original prompt.
    # Comparing the original prompt to each BLIP-generated caption
    # Using ROUGE-L for word overlap
    # Using BERTScore F1 for semantic similarity
    def compute_similarity_scores(self, prompt, captions):
        # Rouge-L
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [rouge.score(prompt, captions)['rougeL'].fmeasure for captions in captions]

        # Bert score
        P, R, F1 = bert_score.score(captions, [prompt] * len(captions), lang="en", rescale_with_baseline=True)

        return {
            "avg_rougeL": sum(rouge_scores) / len(rouge_scores),
            "avg_bertscore_f1": F1.mean().item()
        }
