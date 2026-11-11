# 📧 Email Spam Detection Using Machine Learning

## 📌 Project Overview

This project uses Machine Learning and Natural Language Processing (NLP) to detect whether an email is **Spam** or **Ham (Legitimate)**.
Email text is cleaned, converted to TF-IDF features, and classified with Logistic Regression. A Streamlit app lets users paste an email and get a spam probability.

## ⚠️ Important: Dataset Bias (and how it is handled)

The first version of this project scored 97-98% but marked many normal emails as spam. The cause was **not class imbalance**
(ham is the majority class, about 71%). The cause was **dataset bias**: every ham email in the Enron dataset is an internal
Enron business email (2000-2002), while spam comes from the whole internet. The model learned
*"Enron-style = ham, anything else = spam"* by using words like `enron`, `daren`, `hpl`, `sitara` and years such as `2000`.

Fixes applied in this version:

| Problem | Fix |
|---|---|
| Model memorised Enron-specific words, names and dates | Masked in `text_utils.py` (`SOURCE_WORDS`, month/weekday/year removal) |
| Ham came from one source only | Added `extra_emails.csv` (everyday ham + modern spam/phishing) with higher sample weight |
| Test set looked like the training set | Separate `external_test.csv`, never used for training |
| Too many good emails marked as spam | Logistic Regression probabilities + spam threshold (default 0.7) |
| Training and app cleaned text differently (risk) | One shared `clean_text` in `text_utils.py` used by both |

On the hand-written external test, the old approach marked about 12 of 30 normal emails as spam; the fixed model marks 1 of 30.
This external set is small (45 emails), so treat it as a sanity check and **not** as exact real-world accuracy.

## 📂 Datasets

| File | Purpose |
|---|---|
| `spam_ham_dataset.csv` | Enron Spam/Ham dataset (5,171 emails, 4,993 after removing duplicates) |
| `extra_emails.csv` | 75 ham + 30 spam extra training emails (university, shopping, OTP, family, phishing, lottery ...) |
| `external_test.csv` | 30 ham + 15 spam emails used for testing only |

Labels: `0` = Ham, `1` = Spam.

## 🔄 Workflow

1. Load and inspect data, remove duplicates
2. Show the dataset bias with numbers
3. Clean text with `text_utils.clean_text`
4. Split Enron data 80/20 (stratified)
5. Train the old baseline to demonstrate the problem
6. Train the fixed model (Enron train split + extra emails)
7. Choose the spam threshold
8. Evaluate on the Enron test split and the external test set
9. Inspect the most important words
10. Save `spam_model.pkl` and run the Streamlit app

## 📁 Project Structure

```text
Email-Spam-Detection/
│
├── Email_Spam_Detection.ipynb
├── text_utils.py          # shared text cleaning (used by notebook AND app)
├── app.py
├── spam_model.pkl
├── spam_ham_dataset.csv
├── extra_emails.csv
├── external_test.csv
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Email-Spam-Detection
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 🚀 Run

1. Open `Email_Spam_Detection.ipynb` and run all cells. This regenerates `spam_model.pkl` for your own scikit-learn version.
2. Start the app:

```bash
streamlit run app.py
```

> If you move or rename `text_utils.py`, keep it in the same folder as `app.py`. The app imports it.

## 📈 Evaluation

Precision, recall, F1-score, confusion matrices and a threshold table are reported. For spam filtering, a good email wrongly marked as spam
is usually worse than a missed spam, so the threshold is set above 0.5. You can change it in the app under *Advanced settings*.

## 🔮 Future Improvements

- Add many more real emails from different sources (SpamAssassin, TREC, your own mailbox) and re-run the notebook.
- Use a larger external test set from real emails, not hand-written ones.
- Try sender, header and link features in addition to text.

## ⚠️ Limitations

- Even after masking, some Enron-style words (names such as `bob`, words such as `plant`) may still lean towards ham.
- The extra and external emails are small and hand-written; they reduce the bias but do not remove it.
- The model reads email text only. It does not inspect attachments, sender identity or real link destinations.
- Predictions are not a guarantee that an email is safe.

## 👨‍💻 Author

**Uzair Khan**

BS Computer Science | Artificial Intelligence Specialization

---

**Built with Python, NLP, and Machine Learning.**
