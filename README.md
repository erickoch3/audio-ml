Below is an **updated** `README.md` that explains **each major section** of the repository, **in the recommended order of work**, and how each part **builds on** the previous stages. Feel free to adjust any wording to match your specific workflow or preferences.

---

# Voice Cloning Project

This repository implements a **research-focused** pipeline for **voice cloning**, demonstrating:

1. **Speaker Encoding**: Deriving an embedding that captures a speaker’s unique vocal traits.  
2. **Text-to-Spectrogram Synthesis**: Converting text + speaker embedding into a mel spectrogram.  
3. **Vocoder-Based Waveform Generation**: Turning the mel spectrogram into raw audio.

We also integrate **MLFlow** and **Weights & Biases (wandb)** for **experiment tracking** and reproducibility.

---

## Repository Sections (Recommended Work Order)

Below is the suggested order in which you work through the project’s sections, with an explanation of how each step depends on the last. This structure helps you incrementally **build** and **test** your voice cloning system.

---

### 1. Environment and Dependencies

- **Files**:  
  - `requirements.txt` or `environment.yml`  
  - `Makefile`  
  - `.gitignore`  

- **Purpose**:  
  - Ensures all team members (and your future self) can **recreate** the exact environment.  
  - The **Makefile** provides convenient commands to set up a virtual environment (`.venv`), install or update dependencies, and run scripts.

- **What to Do**:  
  1. Run `make create-env` to create a Python virtual environment.  
  2. Run `make install-requirements` to install the required packages.  

**Why This Matters**: A consistent environment avoids “works on my machine” issues and ensures all subsequent steps (data processing, model training) run smoothly.

---

### 2. Data

- **Folders**:  
  - `data/raw/`  
  - `data/processed/`

- **Purpose**:  
  - Store **raw audio** (e.g., `.wav` files) and any transcripts in `data/raw/`.  
  - After processing (feature extraction, alignment), outputs go in `data/processed/`.

- **What to Do**:  
  1. Collect or record enough high-quality audio samples of the speaker(s) you want to clone.  
  2. Place them in `data/raw/`.  

**How It Builds On Previous Step**:  
- You need the environment set up (step 1) so you can run the data preprocessing scripts.

---

### 3. Data Processing

- **Folder**:  
  - `src/data_processing/`  

- **Key Files**:  
  - `preprocessing.py`  
  - `augmentation.py`  
  - `alignment.py`  

- **Purpose**:  
  1. **Preprocessing**: Convert raw audio to consistent formats (e.g., `.wav` @ 16kHz), extract mel spectrograms, trim silence, etc.  
  2. **Augmentation**: Optionally apply transformations (pitch/time shifting, noise injection) to boost model robustness.  
  3. **Alignment**: (Advanced) Use forced alignment techniques to match phonemes to time segments, helpful for debugging TTS alignment.  

- **What to Do**:  
  1. Update or customize `preprocessing.py` to meet your dataset’s needs.  
  2. Run `make data-preprocess` (or `./scripts/run_data_preprocessing.sh`) to generate mel spectrograms and transcripts in `data/processed/`.  

**How It Builds On Previous Steps**:  
- Preprocessing uses the data you placed in `data/raw/` (step 2).  
- Relies on the libraries installed in your environment (step 1).  
- Outputs become input features for model training in the next stages.

---

### 4. Speaker Encoder

- **Folder**:  
  - `src/speaker_encoder/`  

- **Key Files**:  
  - `encoder_model.py`  
  - `encoder_train.py`  
  - `encoder_inference.py`  

- **Purpose**:  
  - Learn a **latent embedding** that captures speaker identity (pitch, timbre, etc.).  
  - This embedding will be passed to the synthesizer so it can generate speech **in the target speaker’s voice**.

- **What to Do**:  
  1. Review or customize `encoder_model.py` to define your network (e.g., LSTM, CNN).  
  2. Run `make train-encoder` (or `scripts/run_encoder_training.sh`) to train.  
  3. Track and visualize your results in MLFlow or wandb (covered in step 9).  

**How It Builds On Previous Steps**:  
- Uses the **processed spectrogram features** from step 3.  
- Depends on your environment + requirements (step 1).  
- Produces an embedding (model checkpoint) used by the synthesizer next.

---

### 5. Synthesizer (Text-to-Spectrogram)

- **Folder**:  
  - `src/synthesizer/`  

- **Key Files**:  
  - `synthesizer_model.py`  
  - `synthesizer_train.py`  
  - `synthesizer_inference.py`  

- **Purpose**:  
  - A **sequence-to-sequence** model (e.g., Tacotron) that converts **text** and **the speaker embedding** into **mel spectrogram** frames.  
  - The **attention** mechanism aligns input text tokens with acoustic frames.

- **What to Do**:  
  1. Edit `synthesizer_model.py` for your chosen TTS architecture.  
  2. Run `make train-synthesizer` (or `scripts/run_synthesizer_training.sh`) to train on processed data.  
  3. Inspect alignment plots, early reconstructions of spectrograms, etc.

**How It Builds On Previous Steps**:  
- Consumes the **speaker embeddings** from step 4 (the trained encoder).  
- Also uses the **processed data** (spectrograms, transcripts) from step 3 as training references.  
- The output is a “mel spectrogram” that will feed into the vocoder next.

---

### 6. Vocoder

- **Folder**:  
  - `src/vocoder/`  

- **Key Files**:  
  - `vocoder_model.py`  
  - `vocoder_train.py`  
  - `vocoder_inference.py`  

- **Purpose**:  
  - Converts mel spectrograms **(from the synthesizer)** into a **raw waveform**.  
  - Could be an autoregressive model (WaveNet), flow-based (WaveGlow), or GAN-based (HiFi-GAN).

- **What to Do**:  
  1. Implement or select an existing vocoder architecture in `vocoder_model.py`.  
  2. Run `make train-vocoder` (or `scripts/run_vocoder_training.sh`).  
  3. Check generated audio samples for clarity, artifacts, or unnaturalness.

**How It Builds On Previous Steps**:  
- The **synthesizer** (step 5) outputs mel spectrograms that the vocoder needs to produce final audio.  
- Training the vocoder typically requires **paired (spectrogram, audio) data** from step 3’s output.  
- The final output is audio that (hopefully) matches your voice’s timbre and intonation.

---

### 7. Evaluation

- **Folder**:  
  - `src/evaluation/`  

- **Key Files**:  
  - `subjective_eval.py`  
  - `objective_eval.py`  

- **Purpose**:  
  1. **Subjective Evaluation**: Gather **human** feedback via Mean Opinion Score (MOS), ABX tests for speaker similarity, etc.  
  2. **Objective Evaluation**: Use metrics like **Mel Cepstral Distortion (MCD)** or **Word Error Rate (WER)** to gauge clarity/intelligibility.

- **What to Do**:  
  1. After the vocoder produces audio, run `subjective_eval.py` or `objective_eval.py` to measure performance.  
  2. Compare different training setups, hyperparameters, or data augmentation strategies.

**How It Builds On Previous Steps**:  
- Relies on **final audio outputs** from the vocoder to evaluate quality.  
- Also uses reference audio or text for metrics (MCD, WER).  
- Provides **feedback** on whether you need to revisit earlier steps (more data, different model architectures, etc.).

---

### 8. Inference / End-to-End Demo

- **Folder**:  
  - `src/inference/`  

- **Key Files**:  
  - `text_to_speech.py`  
  - `realtime_demo.py`  

- **Purpose**:  
  - A **high-level pipeline** that ties encoder → synthesizer → vocoder together for a single text-to-speech function.  
  - Optionally demonstrates **real-time** capabilities or a user interface (if hardware allows).

- **What to Do**:  
  1. Point `text_to_speech.py` to your **trained models** (encoder, synthesizer, vocoder).  
  2. Input text plus a reference audio sample for the speaker embedding.  
  3. Listen to the final, fully cloned voice output.

**How It Builds On Previous Steps**:  
- It’s the **culmination** of your entire pipeline (steps 3–7).  
- Ensures all the models can run **in sequence** to produce a single final audio result.

---

### 9. Experiment Tracking (MLFlow & wandb)

- **Folder**:  
  - `src/tracking/`  

- **Key Files**:  
  - `mlflow_utils.py`  
  - `wandb_utils.py`  

- **Purpose**:  
  1. **MLFlow**: Track runs, parameters, metrics, and artifacts (model checkpoints).  
  2. **Weights & Biases**: Additional tracking, visualizations, hyperparameter sweeps.  

- **What to Do**:  
  1. Start the MLFlow server (`make run-mlflow-server`) if you want a local UI.  
  2. Configure wandb (`wandb init`) if desired and run sweeps to optimize hyperparameters.  

**How It Builds On Previous Steps**:  
- **Any model training** (encoder, synthesizer, vocoder) can be wrapped in MLFlow or wandb logs.  
- Helps you systematically **compare** experiments (e.g., different learning rates, architectures) across the entire pipeline.

---

### 10. Scripts

- **Folder**:  
  - `scripts/`  

- **Key Files**:  
  - `run_data_preprocessing.sh`  
  - `run_encoder_training.sh`  
  - `run_synthesizer_training.sh`  
  - `run_vocoder_training.sh`  
  - `run_mlflow_server.sh`  
  - `run_wandb_sweep.sh`  

- **Purpose**:  
  - Each script **wraps** the commands for a specific stage (data preprocessing, model training, or launching servers).  
  - Called by the **Makefile** to keep your workflow consistent and easy to remember (`make data-preprocess`, `make train-encoder`, etc.).

**How It Builds On Previous Steps**:  
- Automates the commands you’d otherwise type manually, referencing the Python scripts in `src/`.  
- Ensures consistent usage of your **virtual environment** and **config files** across the pipeline.

---

### 11. Notebooks

- **Folder**:  
  - `notebooks/`  

- **Key Files**:  
  - `00_exploratory_data_analysis.ipynb`  
  - Any other experiment or prototyping notebooks  

- **Purpose**:  
  - **Interactive** exploration of data, quick tests with small subsets of audio, or rapid prototyping of new ideas.  
  - Great for **visualizing** spectrograms, alignment plots, or embedding clusters.

**How It Builds On Previous Steps**:  
- You can do preliminary data checks or network prototypes **before** committing to writing full Python modules.  
- Often used to **debug** or visualize outputs from the pipeline.

---

### 12. Configurations

- **Folder**:  
  - `configs/`  

- **Key Files**:  
  - `encoder_config.yaml`  
  - `synthesizer_config.yaml`  
  - `vocoder_config.yaml`  
  - `wandb_config.yaml`  

- **Purpose**:  
  - **Centralize** hyperparameters (batch size, learning rate, architecture details) for each module.  
  - Keep your code simpler by reading from config files, so you’re not hardcoding values.

**How It Builds On Previous Steps**:  
- Each training script references a config file.  
- When you change configurations, you can easily re-run experiments and track changes in MLFlow/wandb.

---

## Suggested Workflow Recap

1. **Environment Setup** → 2. **Collect Data** → 3. **Data Preprocessing** → 4. **Train Speaker Encoder** → 5. **Train Synthesizer** → 6. **Train Vocoder** → 7. **Evaluate** (subjective & objective) → 8. **Inference** / Demo → 9. **Track Experiments** in MLFlow / wandb → 10. **Use Scripts** to automate steps → 11. **Use Notebooks** for exploration → 12. **Tweak Configs** to refine results.

This **modular approach** ensures each stage feeds neatly into the next, giving you plenty of room for **experimentation** (e.g., swapping out vocoder architectures) while keeping a clear end-to-end pipeline.

---

## License and Ethical Considerations

- **Voice Cloning** can raise ethical issues regarding consent and privacy.  
- Ensure you **own** or have **permission** to use the voice samples in `data/raw/`.  
- Respect local laws and platform policies regarding synthetic media.

---

## Contributing

If you’d like to **contribute** improvements, new architectures, or bugfixes:

1. Fork the repo.  
2. Create a feature branch.  
3. Open a pull request explaining your changes.

---

## Contact

For questions or feedback, please reach out to [Your Name or Email].  

Happy Researching & Cloning!