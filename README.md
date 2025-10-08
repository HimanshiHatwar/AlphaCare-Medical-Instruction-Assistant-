# AlpaCare Medical Instruction Assistant

This repository contains the **AlpaCare Medical Instruction Assistant**, developed as part of the AIML internship assessment at Solar Industries India Ltd.  
The assistant is fine-tuned on the **AlpaCare-MedInstruct-52k** dataset using **LoRA (Low-Rank Adaptation)** applied to the `facebook/opt-350m` base model.  
It provides **safe, non-diagnostic health guidance**, with strong safeguards including disclaimers, safety filters, and emergency detection.



##  Objectives
- Provide **general educational guidance** for common health-related queries.
- Ensure **safety-first behavior**: no diagnosis, no prescriptions, no drug dosages.
- Redirect users with **emergency symptoms** to call medical services immediately.
- Improve clarity with **bullet-point responses** and **disclaimers**.



##  Model & Dataset
- **Base Model:** `facebook/opt-350m` (chosen for Colab compatibility and efficient training)  
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation) using Hugging Face PEFT  
- **Dataset:** [lavita/AlpaCare-MedInstruct-52k](https://huggingface.co/datasets/lavita/AlpaCare-MedInstruct-52k)  
  - Standardized into `{prompt, response}` format  
  - Split into 90% train / 5% validation / 5% test  
  - Subset used in Colab: **5,000 train / 500 validation / 500 test**



##  Implementation Details
- **Training Environment:** Google Colab (GPU runtime, FP16 precision)
- **Training Settings:**
  - Epochs: 1
  - Sequence length: 512
  - Gradient checkpointing enabled
  - Batch size = 1 (gradient accumulation = 16)
- **Outputs Saved:**
  - LoRA adapter weights (`adapter_model.safetensors`)
  - Tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`, etc.)

### Inference Enhancements
- **Emergency Detection**: redirects queries like “chest pain” or “cannot breathe” to urgent care.
- **Safety Filter**: blocks dosage/prescription-related responses.
- **Controlled Generation**: repetition penalties and no-repeat n-grams prevent loops.
- **Text Cleaning**: removes duplicated sentences and trims noisy outputs.
- **Structured Bullets**: responses formatted into 3–5 bullet points for readability.



##  Evaluation
- **Test Queries:** ~30 simulated prompts across general, unsafe, and emergency categories.
- **Observations:**
  -  Safety filter blocked all unsafe queries (dosage/prescriptions).
  -  Emergency detection triggered correctly for urgent symptoms.
  -  Disclaimers consistently appended.
  -  Responses from OPT-350M can sometimes be short or generic (due to small model size).
- **Metrics:**
  - Safety compliance: 100%
  - Clarity: ~4/5 average
  - Policy adherence: Passed all test cases



##  Repository Structure
```
alphaCare/
├── colab_finetune_inference.ipynb   # Training + inference notebook (main file)
├── data_loader.py                   # Dataset preparation script
├── requirements.txt                 # Required Python libraries
├── REPORT.pdf                       # Final technical report (8–9 pages)
└── README.md                        # This file
```



##  Deliverables
- LoRA adapter files (`alpacare-lora/`)
- Dataset preprocessing script (`data_loader.py`)
- Training + inference notebook (`colab_finetune_inference.ipynb`)
- Technical report (`REPORT.pdf`)
- README with setup instructions



##  Future Work
- Extend to **multilingual support** for Indian languages.
- Integrate **citation retrieval** from trusted health sources.
- Add **confidence scoring** and **red-flag symptom alerts**.
- Explore **quantization** to fine-tune larger models (e.g., OPT-1.3B) within Colab resources.



##  Disclaimer
This assistant is for **educational purposes only**.  
It does **not** provide diagnosis, prescriptions, or dosages.  
Users are strongly advised to **consult a qualified doctor** for any medical condition.
