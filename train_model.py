{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4a631edc-113c-4b5f-9606-5c02b20ca7f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9842984409799554\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import re\n",
    "import string\n",
    "import pickle\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# Load datasets\n",
    "fake = pd.read_csv(\"Fake.csv\")\n",
    "true = pd.read_csv(\"True.csv\")\n",
    "\n",
    "# Labels\n",
    "fake[\"label\"] = 0\n",
    "true[\"label\"] = 1\n",
    "\n",
    "# Merge datasets\n",
    "df = pd.concat([fake, true], axis=0)\n",
    "\n",
    "# Shuffle\n",
    "df = df.sample(frac=1, random_state=42)\n",
    "\n",
    "# Combine title and text\n",
    "df[\"content\"] = df[\"title\"] + \" \" + df[\"text\"]\n",
    "\n",
    "# Cleaning function\n",
    "def clean_text(text):\n",
    "    text = str(text).lower()\n",
    "    text = re.sub(r\"http\\\\S+\", \"\", text)\n",
    "    text = re.sub(r\"\\\\d+\", \"\", text)\n",
    "    text = text.translate(str.maketrans(\"\", \"\", string.punctuation))\n",
    "    return text\n",
    "\n",
    "df[\"content\"] = df[\"content\"].apply(clean_text)\n",
    "\n",
    "# Features and labels\n",
    "X = df[\"content\"]\n",
    "y = df[\"label\"]\n",
    "\n",
    "# Train Test Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y,\n",
    "    test_size=0.2,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "# TF-IDF\n",
    "vectorizer = TfidfVectorizer(stop_words=\"english\", max_df=0.7)\n",
    "\n",
    "X_train_tfidf = vectorizer.fit_transform(X_train)\n",
    "X_test_tfidf = vectorizer.transform(X_test)\n",
    "\n",
    "# Model\n",
    "model = LogisticRegression()\n",
    "\n",
    "model.fit(X_train_tfidf, y_train)\n",
    "\n",
    "# Prediction\n",
    "y_pred = model.predict(X_test_tfidf)\n",
    "\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "\n",
    "print(\"Accuracy:\", accuracy)\n",
    "\n",
    "# Save model\n",
    "pickle.dump(model, open(\"models/fake_news_model.pkl\", \"wb\"))\n",
    "pickle.dump(vectorizer, open(\"models/vectorizer.pkl\", \"wb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b4f1ca9b-6475-43c8-8992-d34c67dce491",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "for root, dirs, files in os.walk(\".\"):\n",
    "    for file in files:\n",
    "        if file.endswith(\".pkl\"):\n",
    "            print(os.path.join(root, file))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "5d0f6216-46b2-425e-80a3-33c32f44c77d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Files saved successfully!\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import pickle\n",
    "\n",
    "os.makedirs(\"models\", exist_ok=True)\n",
    "\n",
    "pickle.dump(model, open(\"models/fake_news_model.pkl\", \"wb\"))\n",
    "pickle.dump(vectorizer, open(\"models/vectorizer.pkl\", \"wb\"))\n",
    "\n",
    "print(\"Files saved successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3f32e5c-a829-413b-9bcf-210ebbe43412",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
