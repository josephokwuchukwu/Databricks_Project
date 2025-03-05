# Databricks_Project
import pandas as pd  
data = pd.read_csv('https://s3.amazonaws.com/talent-assets.datacamp.com/recipe_site_traffic_2212.csv')
print(data.head())




import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('https://s3.amazonaws.com/talent-assets.datacamp.com/recipe_site_traffic_2212.csv')  
print("Data loaded successfully.")


print("\nData Information:")
print(data.info())  

print("\nUnique values in 'category':")
print(data['category'].unique())  

print("\nUnique values in 'high_traffic':")
print(data['high_traffic'].unique()) 

 'high_traffic' 
if len(data['high_traffic'].unique()) < 2:
    raise ValueError("The target variable 'high_traffic' needs to have at least two classes.")


high_traffic
data['high_traffic'] = data['high_traffic'].apply(lambda x: 1 if x == 'High' else 0)


label_encoder = LabelEncoder()
data['category'] = label_encoder.fit_transform(data['category'])


def convert_to_numeric(value):
    try:
        return float(value)
    except ValueError:
       
        import re
        match = re.search(r'\d+', value)
        if match:
            return float(match.group(0))
        else:
            return np.nan  

numeric_columns = ['calories', 'carbohydrate', 'sugar', 'protein', 'servings']
for col in numeric_columns:
    data[col] = data[col].apply(convert_to_numeric)


data = data.dropna()


X = data.drop(columns=['recipe', 'high_traffic'])
y = data['high_traffic']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)


y_pred_baseline = baseline_model.predict(X_test)
print("\nBaseline Model Accuracy:", accuracy_score(y_test, y_pred_baseline))
print("Baseline Model Classification Report:\n", classification_report(y_test, y_pred_baseline))


comparison_model = RandomForestClassifier(random_state=42)
comparison_model.fit(X_train, y_train)


y_pred_comparison = comparison_model.predict(X_test)
print("\nComparison Model Accuracy:", accuracy_score(y_test, y_pred_comparison))
print("Comparison Model Classification Report:\n", classification_report(y_test, y_pred_comparison))


print("\nModel Comparison:")
print("Baseline Model Accuracy:", accuracy_score(y_test, y_pred_baseline))
print("Comparison Model Accuracy:", accuracy_score(y_test, y_pred_comparison))


plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_comparison), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Comparison Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

