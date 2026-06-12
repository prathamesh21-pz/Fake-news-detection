{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d7577170-db14-4ccd-a99e-c9edbd8375cf",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-06-12 15:33:01.835 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\prath\\AppData\\Roaming\\Python\\Python312\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import re\n",
    "import string\n",
    "\n",
    "# Load model\n",
    "model = pickle.load(open(\"models/fake_news_model.pkl\", \"rb\"))\n",
    "vectorizer = pickle.load(open(\"models/vectorizer.pkl\", \"rb\"))\n",
    "\n",
    "def clean_text(text):\n",
    "    text = text.lower()\n",
    "    text = re.sub(r\"http\\\\S+\", \"\", text)\n",
    "    text = text.translate(str.maketrans(\"\", \"\", string.punctuation))\n",
    "    return text\n",
    "\n",
    "st.title(\"📰 Fake News Detection System\")\n",
    "\n",
    "st.write(\"Enter a news article below\")\n",
    "\n",
    "news = st.text_area(\"News Text\")\n",
    "\n",
    "if st.button(\"Check News\"):\n",
    "\n",
    "    cleaned = clean_text(news)\n",
    "\n",
    "    vector = vectorizer.transform([cleaned])\n",
    "\n",
    "    prediction = model.predict(vector)\n",
    "\n",
    "    if prediction[0] == 1:\n",
    "        st.success(\"✅ Real News\")\n",
    "    else:\n",
    "        st.error(\"❌ Fake News\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c3d9d1ec-dbdd-4c83-8fa0-14a8e629a75a",
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
