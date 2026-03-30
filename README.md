Finished project on 3/29/26.

Results on 2500 unseen set:
ROC-AUC: 0.8343790386703789
              precision    recall  f1-score   support

          No       0.89      0.75      0.81      1851
         Yes       0.53      0.76      0.63       691

    accuracy                           0.75      2542
   macro avg       0.71      0.76      0.72      2542
weighted avg       0.80      0.75      0.76      2542

Business Translation:
Out of 2542 customers:
      You would target for outreach: ~991 customers (~39%)
            Of those: ~525 will actually churn
            ~466 are false alarms
      You miss:
      ~166 churners (~76% caught)
Judgement: Model is tuned and suitable for low-cost outreach.

Top 3 Churn Drivers:

      feature        importance
      
num__TotalCharges     0.152004

num__MonthlyCharges   0.135583

num__tenure           0.133076

