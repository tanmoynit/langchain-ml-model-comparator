Iris ML Algorithm Selector (with LangChain)

A small, self-contained Python project that:

Loads the classic Iris dataset (built into scikit-learn — no download needed).
Trains 7 different machine learning classification algorithms.
Validates each one using Stratified 5-Fold Cross Validation.
Ranks the algorithms by mean accuracy (plus F1, precision, recall, and training time).
Uses LangChain to turn that results table into a short, plain-English recommendation of which algorithm is best for the dataset.
Can be pointed at your own CSV dataset instead of Iris, with a single command-line flag — no code changes required.

Everything runs fully offline by default. LangChain is used for structured prompt-building either way; if you add an OpenAI API key later, the exact same prompt gets sent to a real LLM instead of the built-in offline formatter — see Optional: real LLM mode below.

Project structure
iris_ml_selector/
├── main.py                  # Entry point — run this
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/
│   └── sample_custom_dataset.csv   # Example of a "bring your own dataset" CSV (Wine dataset)
├── output/                  # Created automatically — results.csv & report.txt land here
└── src/
    ├── data_loader.py       # Loads Iris OR any custom CSV
    ├── models.py             # Defines the candidate ML algorithms
    ├── evaluator.py           # Runs Stratified 5-Fold Cross Validation
    └── langchain_report.py    # Builds the LangChain prompt & generates the recommendation


How it works (pipeline overview)
data_loader.py  --->  models.py  --->  evaluator.py  --->  langchain_report.py  --->  console + output/
 (load X, y)         (7 candidate      (Stratified 5-fold      (LangChain prompt ->
                       algorithms)       CV, scoring)             recommendation text)
Algorithms compared (src/models.py)
Model	Notes
Logistic Regression	scaled with StandardScaler inside a pipeline
Support Vector Machine (RBF kernel)	scaled
K-Nearest Neighbors	scaled
Decision Tree	no scaling needed
Random Forest	no scaling needed
Gradient Boosting	no scaling needed
Gaussian Naive Bayes	no scaling needed

Scaling is done inside a scikit-learn Pipeline for the models that need it, so the scaler is fit only on each fold's training data — this avoids data leakage from the validation fold into training, which is a common cross-validation mistake.

Validation method (src/evaluator.py)

Uses StratifiedKFold(n_splits=5, shuffle=True, random_state=42) with scikit-learn's cross_validate. Stratified K-Fold keeps the class proportions consistent in every fold, which is important for classification tasks. You can change the number of folds with --folds (see below).

LangChain layer (src/langchain_report.py)

A langchain_core.prompts.PromptTemplate is built from the results table and asks for: the best model + accuracy, how consistent it was across folds, how it compares to the runner-up, any notably fast/slow models, and a one-line recommendation.

If OPENAI_API_KEY is set (and langchain-openai is installed), this prompt is sent to a real LLM (gpt-4o-mini by default) through a LangChain PromptTemplate | ChatOpenAI chain.
If no API key is set (the default), the same information is rendered by a local, deterministic formatter — so the project works immediately on any standalone machine with no internet access or paid API key required.
Requirements
Python 3.9+
pip

All Python dependencies are listed in requirements.txt:

scikit-learn, pandas, numpy — data handling & ML models
langchain, langchain-core — prompt orchestration
langchain-openai — optional, only needed for real LLM mode
python-dotenv — optional, convenient way to load an API key from a .env file
tabulate — used by pandas for pretty console tables
Setup on a standalone computer (step by step)

These steps assume a fresh machine with nothing installed except Python.

1. Get the project files

Copy/unzip the iris_ml_selector/ folder onto the machine (e.g. via USB drive, git clone, or file transfer — no internet required for this step).

2. (Recommended) Create a virtual environment

This keeps the project's dependencies isolated from the rest of the system.

macOS / Linux:

bash
cd iris_ml_selector
python3 -m venv venv
source venv/bin/activate

Windows (PowerShell):

powershell
cd iris_ml_selector
python -m venv venv
venv\Scripts\Activate.ps1
3. Install dependencies
bash
pip install -r requirements.txt

(If you don't plan to ever use real LLM mode, you can skip installing langchain-openai — everything else still works.)

4. Run the project
bash
python main.py

That's it. You should see:

Dataset summary (150 Iris samples, 4 features, 3 classes)
A ranked results table across all 7 algorithms
The best-performing model
A plain-English recommendation report
Confirmation that output/results.csv and output/report.txt were saved
Using YOUR OWN dataset instead of Iris

No code changes are required. Use the --csv and --target flags:

bash
python main.py --csv path/to/your_data.csv --target your_label_column_name

Optional flags:

bash
python main.py \
  --csv data/sample_custom_dataset.csv \
  --target wine_class \
  --dataset-name "Wine Quality" \
  --folds 10 \
  --output-dir output_wine
Flag	Meaning	Default
--csv	Path to your CSV file	(none — uses built-in Iris)
--target	Name of the column containing the class label	(required if --csv is set)
--dataset-name	Friendly name shown in the report	Iris, or your CSV's filename
--folds	Number of cross-validation folds	5
--output-dir	Where to save results.csv / report.txt	output

Notes on custom CSVs:

Your CSV should have one row per sample, with feature columns + one label/target column.
Non-numeric feature columns are automatically dropped (with a warning printed to the console) since the demo models expect numeric input — encode categorical features to numbers first if you need them included.
The target column can contain either strings (e.g. "setosa") or numbers — it's automatically label-encoded.

A ready-made example is included: data/sample_custom_dataset.csv (the Wine dataset, 178 samples, 13 features, 3 classes). Try it with:

bash
python main.py --csv data/sample_custom_dataset.csv --target wine_class --dataset-name "Wine"
Optional: real LLM mode with LangChain + OpenAI

By default no API key is required. If you'd like the recommendation report to be written by an actual LLM instead of the offline formatter:

Install the optional package (if you skipped it earlier):
bash
   pip install langchain-openai
Set your OpenAI API key as an environment variable: macOS/Linux:
bash
   export OPENAI_API_KEY="sk-..."

Windows (PowerShell):

powershell
   $env:OPENAI_API_KEY="sk-..."
Run python main.py as usual — it will automatically detect the key and route the same LangChain prompt through ChatOpenAI (gpt-4o-mini) instead of the offline formatter.

To use a different LLM provider (Anthropic, local models via Ollama, etc.), swap the ChatOpenAI(...) call in src/langchain_report.py::_try_build_llm_chain() for the equivalent LangChain chat model class — the PromptTemplate and the rest of the pipeline don't need to change.

Extending the project
Add more algorithms: edit src/models.py, add an entry to the dict returned by get_candidate_models().
Change scoring metrics: edit the scoring tuple passed to evaluate_models() in main.py (any scikit-learn scorer name works).
Change number of folds: --folds N on the command line.
Automate a batch of datasets: call data_loader.load_dataset(), models.get_candidate_models(), and evaluator.evaluate_models() directly from your own script — each module is independent and reusable.
Troubleshooting
Problem	Fix
ModuleNotFoundError	Make sure your virtual environment is activated and you ran pip install -r requirements.txt
Target column 'X' not found in ...csv	Check the exact column name in your CSV (case-sensitive) and pass it via --target
All feature columns dropped / "No numeric feature columns remain"	Your CSV's features are all non-numeric (e.g. text categories) — encode them to numbers first (e.g. one-hot or label encoding)
Report says "No OPENAI_API_KEY found"	Expected in offline mode — this is not an error, it's just telling you it used the local formatter instead of a paid LLM call
Slow on a large custom dataset	Reduce --folds, or remove slower models (Gradient Boosting, Random Forest) from src/models.py
