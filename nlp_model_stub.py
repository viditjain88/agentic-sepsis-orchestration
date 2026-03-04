import os
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# This loads the actual Bio_ClinicalBERT model.
# WARNING: Running this model for inference requires significant RAM/VRAM.
# The user intends to run this locally to observe the performance changes.

class FineTunedClinicalBERT:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        print(f"Loading tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading ACTUAL model: {model_name} (This may take a moment and use high memory)...")
        # Load the pre-trained model and add a binary classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode

    def predict_sepsis_probability(self, clinical_note: str) -> float:
        """
        Runs actual inference using the Bio_ClinicalBERT model.
        Returns a probability score [0.0, 1.0] for sepsis presence.
        """
        # 1. Tokenize (as required by Method 3: max 512 subword tokens)
        inputs = self.tokenizer(
            clinical_note,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. Model Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3. Calculate probability using softmax
        # outputs.logits shape is [batch_size, num_labels] -> [1, 2]
        probs = torch.softmax(outputs.logits, dim=-1)

        # We assume class '1' is the positive (sepsis) class
        # Since this model is un-finetuned right out of the box in this script,
        # the weights are random for the classification head, but it executes the true architecture.
        # To see *meaningful* prediction changes, you must provide a path to a checkpoint
        # trained via nlp_finetune.py
        sepsis_prob = probs[0][1].item()

        return sepsis_prob

if __name__ == "__main__":
    # Test the stub
    model = FineTunedClinicalBERT()

    note_pos = "Patient presents with severe hypotension and elevated lactate. Diagnosed with sepsis (ICD-9 995.92)."
    prob_pos = model.predict_sepsis_probability(note_pos)
    print(f"Positive Note Probability: {prob_pos:.2f}")

    note_neg = "Patient recovering well from surgery. Vital signs stable. Discharged home."
    prob_neg = model.predict_sepsis_probability(note_neg)
    print(f"Negative Note Probability: {prob_neg:.2f}")
