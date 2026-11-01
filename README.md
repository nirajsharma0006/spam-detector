# 📩 Indian Spam Detector

Paste an SMS / message — the app tells you whether it's **Normal**, **Spam**, or **Scam**, with a confidence % and a reason.

## What's inside

- **Data:** `data/messages.csv` — 300 realistic Indian messages (English + Roman-Hindi + Hinglish), balanced across the three labels (100 each).
- **Models:** TF-IDF (word 1-2 + char 3-5) with Naive Bayes and Logistic Regression. Both are compared, and the one with the higher macro-F1 is saved.
- **Features:** links, phone numbers, amounts, urgency words, sensitive words (KYC/OTP/PIN), reward words, and excessive CAPS — all detected and shown in the "Why?" section.
- **UI:** Streamlit — text box, Predict button, 3 example buttons, confidence bar chart, model info expander.

## Run commands

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py