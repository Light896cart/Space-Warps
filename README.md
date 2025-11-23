# Space Warps: Progressive Neural Architecture Search (PNAS)

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Machine Learning-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="ML"/>
  <img src="https://img.shields.io/badge/Computer Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="CV"/>
</p>

<p align="center">
  <b>An experimental system for automated construction and training of convolutional neural networks on astronomical data using progressive architecture search.</b>
</p>

## 🎯 What Is This Project About?

This project implements the state-of-the-art **Progressive Neural Architecture Search (PNAS)** method. Rather than relying on a predefined neural network architecture (e.g., ResNet, VGG), the algorithm autonomously decides which layers and with which parameters to add at each step to maximize validation accuracy.

**Task:** Classification of astronomical galaxy and cosmic object images (Space_Galaxi dataset), enhanced with auxiliary features (redshift `z` and its uncertainty `z_err`).

---

## ✨ Key Features & Technologies

*   **Progressive Architecture Search (PNAS):** The algorithm incrementally grows the model, evaluating multiple candidate blocks at each step and selecting the best-performing one.
*   **Warm-Start Weight Initialization (Weight Copying & Jitter):** When adding new blocks, weights from the best candidate of the previous step are copied, scaled, and perturbed with noise—accelerating convergence and improving final performance.
*   **Multimodal Input Support:** The model jointly processes images and tabular features (`z`, `z_err`), projecting them into a unified feature space before classification.
*   **Modular & Reproducible Pipeline:** The code is cleanly structured into modules (datasets, augmentation, model, training, logging), ensuring readability and easy extensibility.
*   **Full Reproducibility:** All random seeds (dataloaders, dataset split, training) are fixed, guaranteeing bit-for-bit reproducibility of any experiment.

---

## 🏗️ System Architecture

### 1. Model (`ProgressiveModel`)
*   **Dynamic Construction:** Built with `nn.ModuleList`, enabling flexible runtime addition of blocks.
*   **Block Design:** Each block follows the sequence `Conv2d → BatchNorm → ReLU → Dropout2d → MaxPool2d`.
*   **Multimodality:** Auxiliary features pass through a dedicated `nn.Sequential` projector and are concatenated with image-derived features before the classifier.
*   **Automatic Head Adaptation:** The classifier head is rebuilt automatically when a new block is added, adapting to the updated feature map size.

### 2. Search Algorithm (`progressive_architecture_search`)
The algorithm proceeds iteratively (up to `max_layers`):
1.  **Initialization:** Starts from an empty base model.
2.  **Candidate Evaluation:** For each `k` in `k_candidates`:
    *   A candidate block is appended to a copy of the current model.
    *   Fine-tuning is performed for `epochs_per_eval` epochs.
    *   Validation accuracy is recorded.
3.  **Best Selection:** The candidate with the highest validation accuracy is permanently added.
4.  **Weight Warm-Start:** Weights of the selected block are copied into the main model, scaled, and jittered (noise added) to serve as initialization for evaluating *other* candidates at the same depth.
5.  **Early Stopping:** Search halts if adding a new layer yields no accuracy gain.

### 3. Data & Augmentation
*   **`Space_Galaxi` Dataset:** Loads images, labels, and auxiliary features from a CSV annotation file.
*   **Augmentation Pipeline:** Includes affine transforms (rotation, scale, shear), color jitter, and additive noise.
*   **DataLoaders:** `create_train_val_dataloaders` provides reproducible train/val splitting, optional data subsampling (`fraction`), and separate augmentations for train vs. val.

---

## 📁 Repository Structure
```
Space_Warps/
├── src/
│ ├── data/
│ │ ├── dataloader.py # Create DataLoader
│ │ └── dataset.py # Class Dataset Space_Galaxi
│ ├── model/
│ │ ├── model_architecture.py # Class ProgressiveModel
│ │ ├── progressive_search.py  # PNAS algorithm implementation
│ │ └── train_eval.py # Training & evaluation utilities
│ ├── utils/
│ │ ├── logging.py # Logging & plotting
│ │ └── seeding.py # Seed fixing utilities
│ └── preprocessing/
│ └── augmentation.py # Augmentation transforms
├── configs/ # (Optional: Hydra config files)
├── results/ # Output directory for logs, models, plots
├── main.py # Main entrypoint
└── README.md
```

---

## 🚀 How to Run

1.  **Clone & Install Dependencies:**
    ```bash
    git clone <your-repo-url>
    cd Space_Warps
    pip install -r requirements.txt  # torch, torchvision, pandas, matplotlib, hydra-core, omegaconf, tqdm
    ```

2.  **Prepare Data:**
    *   Place images in `data/image_data/img_csv_0001/`.
    *   Ensure the annotation file (`balanced_2001_by_class_cycle.csv`) is located at `data/reg/`.

3.  **Launch the Experiment:**
    ```bash
    python main.py
    ```

4.  **Customization (via Code):** Experiment parameters (paths, batch size, candidates, learning rate, etc.) are easily adjustable inside the `main()` function in `main.py`.

---

## 📊 Results & Insights

*   **Primary Metric:** Validation accuracy.
*   **Visualization:** The script automatically saves `progressive_summary.json` and generates a plot showing accuracy progression vs. architecture depth.
*   **Conclusion:** PNAS discovers compact, high-performance architectures *from scratch*, tailored specifically to the dataset—often outperforming standard fixed-depth networks.

---

## 🛠️ Tech Stack

*   **Language:** Python 3.8+
*   **ML Framework:** PyTorch 1.9+, TorchVision
*   **Data Handling:** Pandas, NumPy, PIL
*   **Visualization:** Matplotlib
*   **Config Management:** Hydra, OmegaConf (prepared, disabled by default)
*   **Utilities:** tqdm

---

## 👨‍💻 Author

**Artem Goncharov**

*   Machine Learning Engineer (Computer Vision / NLP)
*   **GitHub:** [Artem Goncharov](https://github.com/artemgoncharov)
*   **Portfolio:** [Other Projects](https://github.com/artemgoncharov?tab=repositories)

*This project was developed as a deep dive into AutoML and Neural Architecture Search capabilities.*

---

## 🔮 Future Improvements

*   **Hydra Configuration Integration:** Migrate all hyperparameters to `.yaml` config files.
*   **Advanced Experiment Tracking:** Integrate Weights & Biases or TensorBoard.
*   **Diverse Candidate Blocks:** Support Residual, MobileNet, or Inception-style blocks.
*   **Joint Architecture & Hyperparameter Search:** Expand search space to include learning rate, dropout, etc.
*   **Model Deployment:** Wrap final model in a REST API (e.g., using FastAPI).

---
**⭐ If you found this project helpful, please consider giving it a star on GitHub!**
