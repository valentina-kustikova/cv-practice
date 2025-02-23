python main.py --train train --test test --clusters 25 --algorithm sift --classifier svm

Classification Report:
              precision    recall  f1-score   support

         Dog       0.70      0.62      0.66       444
         Cat       0.66      0.73      0.69       445

    accuracy                           0.68       889
   macro avg       0.68      0.68      0.68       889
weighted avg       0.68      0.68      0.68       889

2025-02-20 21:44:50,705 [INFO] Accuracy: 67.60%
Accuracy: 67.60%

python main.py --train train --test test --clusters 40 --algorithm sift --classifier svm

Classification Report:
              precision    recall  f1-score   support

         Dog       0.71      0.64      0.67       444
         Cat       0.67      0.73      0.70       445

    accuracy                           0.69       889
   macro avg       0.69      0.69      0.69       889
weighted avg       0.69      0.69      0.69       889

2025-02-20 22:29:34,688 [INFO] Accuracy: 68.62%