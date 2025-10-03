# SMS Spam Classifier

This is a small machine learning project I did for the LinkPlus AI Internship Challenge.  
The goal is to train a simple model that can classify SMS messages as **ham (normal)** or **spam**.  

---

## Dataset
I used the **SMS Spam Collection** dataset from UCI Machine Learning Repository:  
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection  

After downloading, I put the `SMSSpamCollection` file in the same folder as my code.

---

## Steps I did
1. Loaded and checked the dataset (ham vs spam counts).
2. Preprocessed the messages (lowercase + stopwords removed with TF-IDF).
3. Split the data into train and test sets.
4. Trained a Logistic Regression model.
5. Evaluated it with accuracy, precision, recall, F1-score.
6. Made a confusion matrix for visualization.
7. Wrote a small function so I can test new messages like  
   `"You won a free iPhone!!!"` → spam.

---

## How to run
1. Make sure you have Python installed (I used Python 3.13).  
2. Install the required libraries:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
3. Put the dataset file (SMSSpamCollection) in the same folder as the script.
4. Run the script: python spam_classifier.py

   ```bash
   pip install pandas scikit-learn matplotlib seaborn
  
