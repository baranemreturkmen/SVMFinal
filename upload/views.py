from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score #classification_report

def clean_text(text):
    print("Entered clean_text")
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def process_csv(file):
    print("Entered process_csv")
    data = pd.read_csv(file)
    data = data.sample(10000, random_state=42) 
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    data['review'] = data['review'].apply(clean_text)
    print("Out clean text")

    X_train, X_test, y_train, y_test = train_test_split(
        data['review'], data['sentiment'], test_size=0.2, random_state=42, stratify=data['sentiment']
    )
    print("train tedt split ended")

    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print("tfidf ended")

    svm = SVC(kernel='linear', random_state=42)#probability=True,
    svm.fit(X_train_tfidf, y_train)
    print("svm fit ended")

    y_pred = svm.predict(X_test_tfidf)
    y_pred_proba = svm.decision_function(X_test_tfidf)

    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    print("conf maxrix created")
    #class_report = classification_report(y_test, y_pred, output_dict=True)
    #print("class report created")
    metrics = performance_metrics(y_test, y_pred)
    print("metrics created")
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print("roc auc created")

    return {
        "confusion_matrix": conf_matrix,
        #"classification_report": class_report,
        "performance_metrics": metrics,
        "roc_auc": roc_auc
    }

def performance_metrics(y_true, y_pred):
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metrics Calculations
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)  # True Negative Rate
    f1 = f1_score(y_true, y_pred)

    # Return as a dictionary
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1
    }

def upload_csv(request):
    if request.method == "POST":
        file = request.FILES['file']
        results = process_csv(file)
        return JsonResponse(results)
    return render(request, 'upload.html')