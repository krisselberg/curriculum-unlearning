## ⚡ Quickstart

```bash
# Environment setup
conda create -n unlearning python=3.11
conda activate unlearning
pip install .
pip install --no-build-isolation flash-attn==2.6.3

# Data setup
python setup_data.py  # saves/eval now contains evaluation results of the uploaded models
# Downloads log files with metric eval results (incl retain model logs) from the models 
# used in the supported benchmarks.
```

---

### 📜 Running Experiments

The codebase supports three main types of experiments, each with its own dedicated script:

1. **Curriculum Learning Experiments**
   - Uses a curriculum strategy to order forget examples by difficulty
   - Two strategies available: easy-to-hard and hard-to-easy
   - Config: `configs/experiment/unlearn/tofu/curriculum.yaml`
   - Run: `bash scripts/run_curriculum.sh`

2. **SGD Experiments**
   - Replaces AdamW with vanilla SGD optimizer
   - Config: `configs/experiment/unlearn/tofu/sgd.yaml`
   - Run: `bash scripts/run_sgd.sh`

3. **Learning Rate and Epoch Experiments**
   - Tests different learning rates and number of epochs
   - Config: `configs/experiment/unlearn/tofu/lr_epoch.yaml`
   - Run: `bash scripts/run_lr_epoch.sh`

Each script handles:
- Setting up the appropriate environment variables
- Loading the correct model and dataset splits
- Running the training process with the specified configuration
- Evaluating the trained model

---

## 📝 Citing this work

If you use OpenUnlearning in your research, please cite OpenUnlearning and the benchmarks from the below:

```bibtex
@misc{openunlearning2025,
  title={{OpenUnlearning}: A Unified Framework for LLM Unlearning Benchmarks},
  author={Dorna, Vineeth and Mekala, Anmol and Zhao, Wenlong and McCallum, Andrew and Kolter, J Zico and Maini, Pratyush},
  year={2025},
  howpublished={\url{https://github.com/locuslab/open-unlearning}},
  note={Accessed: February 27, 2025}
}
@inproceedings{maini2024tofu,
  title={{TOFU}: A Task of Fictitious Unlearning for LLMs},
  author={Maini, Pratyush and Feng, Zhili and Schwarzschild, Avi and Lipton, Zachary Chase and Kolter, J Zico},
  booktitle={First Conference on Language Modeling},
  year={2024}
}
@article{shi2024muse,
  title={MUSE: Machine Unlearning Six-Way Evaluation for Language Models},
  author={Weijia Shi and Jaechan Lee and Yangsibo Huang and Sadhika Malladi and Jieyu Zhao and Ari Holtzman and Daogao Liu and Luke Zettlemoyer and Noah A. Smith and Chiyuan Zhang},
  year={2024},
  eprint={2407.06460},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2407.06460},
}
```

### 📄 License
This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

