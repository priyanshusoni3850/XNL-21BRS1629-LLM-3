# XNL Project

![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)

## 🚀 Project Overview
XNL is a comprehensive solution designed to execute machine learning workflows seamlessly. It includes two primary `.ipynb` files for different approaches and a `app.py` script for the backend.

## 📂 Files and Directories

| File Name        | Description |
|-----------------|-------------|
| **`XNL_VERSIO.ipynb`** | **(Recommended)** Separated phase approach, where each phase is executed independently, providing a more detailed explanation, structured outputs, and comprehensive conclusions. |
| `XNL_COLAB.ipynb` | Fully integrated backend in one execution pipeline. Optimized for quick execution. |
| `app.py` | Backend script (may require virtual environment due to dependency clashes). |
| `frontend/` | Simple frontend to fetch and display data from the backend. |

## 💡 How to Use
### 🛠️ Setting Up the Environment
Before running the project, set up a virtual environment to avoid dependency issues:
```bash
python -m venv xnl_env
source xnl_env/bin/activate  # On Mac/Linux
xnl_env\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 📌 Running the Notebooks on Google Colab
1. Open **`XNL_VERSIO.ipynb`** (Recommended) or `XNL_COLAB.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Navigate to `Runtime` > `Change runtime type`.
3. Set **Hardware accelerator** to **T4 GPU**.
4. Run all cells (`Runtime` > `Run all`).

> **Recommended:** `XNL_VERSIO.ipynb` provides a more structured breakdown, stepwise execution, detailed outputs, and well-documented conclusions for better understanding.

### 🖥️ Running the Backend
```bash
python app.py
```
> **Note:** `app.py` might encounter dependency clashes. It's recommended to run it inside a virtual environment.

### 🌐 Running the Frontend
1. Ensure the backend is running.
2. Navigate to the `frontend` directory and open `index.html` in a browser.

## 🎯 Features
✅ **`XNL_VERSIO.ipynb` (Recommended)** - Modular execution with separate phases, offering **detailed insights, structured outputs, and comprehensive conclusions**.
✅ **`XNL_COLAB.ipynb`** - Fully integrated backend in one execution, optimized for quick execution.
✅ **GPU Support** - Works best with Google Colab's **T4 GPU**.
✅ **Virtual Environment Support** - Avoids dependency clashes for backend execution.
✅ **Simple Frontend** - Fetches data from the backend and displays it.

## 🤝 Contributions
Feel free to fork, modify, and contribute! Open a PR for enhancements and fixes.

## 📜 License
This project is licensed under the MIT License. See `LICENSE` for details.

---
🚀 **Happy Coding!**

