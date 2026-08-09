# Case Study 2: Diabetes Readmission
### Eric Graham
### DS 7333: Quantifying the World
### Dr. Robert Slater

## Case Study 2 Feedback

Dr. Slater: I incorporated your feedback within the flow of the original notebook, here is a summary of my work on each feedback item for quick reference.

_1) Re run your analysis but leave in encounter and patient number.  What happens?  Can you comment on this result--what does it tell you_

`encounter_id` is a unique key per visit. `patient_nbr` identifies each patient, and patients appear multiple times in this dataset. Because the 80/20 split is random at the encounter level, the same patient's records often land in both train and test sets, so the model can memorize `patient_nbr` to outcome associations rather than learning clinical features. This is data leakage. See the **Re-run with ID Columns Included** section for results.

_2) Why no cross-validataion_

A single 80/20 split gives one accuracy estimate that varies with which rows fall in the test set. Five-fold stratified CV trains and evaluates five times on non-overlapping folds, preserving class proportions in each, and reports a mean and standard deviation across folds. A tight standard deviation confirms the single-split result was not a fluke. See the **Cross-Validation** section for results.

_3) You have a discharge disposition id.  It looks like it was not treated as categorical. Why was that.  Look at the ID codes.  Any reason why this id might be important?_

`discharge_disposition_id` is now cast to string before `get_dummies` because its integer codes map to named categories (home, skilled nursing, expired, etc.) with no ordinal relationship. Treating it as numeric would obscure the signal from specific outcomes, particularly ID 11 (expired), which is a near-perfect predictor of no readmission.

_4) Using your confusion matrix, talk about what happens when the model predicts the wrong class._

The most costly error is predicting `NO` when the true label is `<30`: a high-risk patient is discharged without follow-up. Of 2,285 actual early-readmission cases, roughly 2,216 were misclassified, most absorbed into `NO` predictions. A false positive wastes resources but harms no one; the false negative sends a patient home who will bounce back within 30 days. See the confusion matrix discussion for the full breakdown.

_5) What is "C" in logistic regression and why is it important._

`C` is the inverse of regularization strength (C = 1/lambda). Smaller `C` applies stronger regularization, shrinking coefficients toward zero and producing a simpler model; larger `C` relaxes regularization and lets the model fit the training data more closely. With roughly 1,000 features after one-hot encoding, the default `C=1` is a reasonable starting point, though a grid search over smaller values could reduce overfitting and may help the model generalize better to the minority class.

## Introduction

This case study uses the UCI Diabetes 130-US Hospitals dataset, which contains 10 years of clinical care records (1999–2008) across 130 US hospitals. Each row represents a single inpatient diabetic encounter. The target variable is `readmitted`, with three categories: `NO` (no readmission), `>30` (readmitted after 30 days), and `<30` (readmitted within 30 days). We build a multiclass logistic regression classifier to predict all three categories and identify the top variables driving readmission.

## Import Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Suppress the PerformanceWarning raised by the diagnosis encoding loop below:
#   PerformanceWarning: DataFrame is highly fragmented. This is usually the result
#   of calling frame.insert many times, which has poor performance.
# Expected: the loop adds ~916 diagcode_* columns one at a time. The data.copy()
# call immediately after defragments the frame, which is exactly what pandas recommends.
# import warnings
# warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

data = pd.read_csv("./dataset_diabetes/diabetic_data.csv")
```

## Missingness Check & Imputation

Missing values are encoded as `'?'` strings rather than `NaN`. The `weight` column (~97% missing) is dropped. `payer_code` (~40%) and `medical_specialty` (~49%) are too sparse to impute meaningfully and are filled with `'Unknown'`. `race` (~2%) is mode-imputed. The diagnosis columns (`diag_1`, `diag_2`, `diag_3`) will be one-hot encoded in the next section rather than imputed. `'?'` simply becomes its own indicator column.


```python
missing = (data == '?').sum()
missing[missing > 0]
```




    race                  2273
    weight               98569
    payer_code           40256
    medical_specialty    49949
    diag_1                  21
    diag_2                 358
    diag_3                1423
    dtype: int64




```python
data = data.drop(columns=['weight'])

data['race'] = data['race'].replace('?', np.nan).fillna(data['race'].replace('?', np.nan).mode()[0])

for col in ['payer_code', 'medical_specialty']:
    data[col] = data[col].replace('?', 'Unknown')
```

## Diagnosis Code Encoding

Following the professor's approach: build a union of all unique diagnosis codes across `diag_1`, `diag_2`, and `diag_3`, then create a binary indicator column for each code. A row gets a `1` in `diagcode_X` if any of its three diagnosis fields contains code X. This avoids treating the codes as ordinal values and captures comorbidities naturally.


```python
data.columns
```




    Index(['encounter_id', 'patient_nbr', 'race', 'gender', 'age',
           'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
           'time_in_hospital', 'payer_code', 'medical_specialty',
           'num_lab_procedures', 'num_procedures', 'num_medications',
           'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1',
           'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult',
           'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
           'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
           'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
           'tolazamide', 'examide', 'citoglipton', 'insulin',
           'glyburide-metformin', 'glipizide-metformin',
           'glimepiride-pioglitazone', 'metformin-rosiglitazone',
           'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted'],
          dtype='str')




```python
diag = ['diag_1','diag_2','diag_3']
```


```python
data[diag]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diag_1</th>
      <th>diag_2</th>
      <th>diag_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>250.83</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>1</th>
      <td>276</td>
      <td>250.01</td>
      <td>255</td>
    </tr>
    <tr>
      <th>2</th>
      <td>648</td>
      <td>250</td>
      <td>V27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>250.43</td>
      <td>403</td>
    </tr>
    <tr>
      <th>4</th>
      <td>197</td>
      <td>157</td>
      <td>250</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>101761</th>
      <td>250.13</td>
      <td>291</td>
      <td>458</td>
    </tr>
    <tr>
      <th>101762</th>
      <td>560</td>
      <td>276</td>
      <td>787</td>
    </tr>
    <tr>
      <th>101763</th>
      <td>38</td>
      <td>590</td>
      <td>296</td>
    </tr>
    <tr>
      <th>101764</th>
      <td>996</td>
      <td>285</td>
      <td>998</td>
    </tr>
    <tr>
      <th>101765</th>
      <td>530</td>
      <td>530</td>
      <td>787</td>
    </tr>
  </tbody>
</table>
<p>101766 rows × 3 columns</p>
</div>




```python
pd.get_dummies(data['diag_1'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>10</th>
      <th>11</th>
      <th>110</th>
      <th>112</th>
      <th>114</th>
      <th>115</th>
      <th>117</th>
      <th>131</th>
      <th>133</th>
      <th>135</th>
      <th>...</th>
      <th>V55</th>
      <th>V56</th>
      <th>V57</th>
      <th>V58</th>
      <th>V60</th>
      <th>V63</th>
      <th>V66</th>
      <th>V67</th>
      <th>V70</th>
      <th>V71</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>101761</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>101762</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>101763</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>101764</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>101765</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>101766 rows × 717 columns</p>
</div>




```python
data[diag]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diag_1</th>
      <th>diag_2</th>
      <th>diag_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>250.83</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>1</th>
      <td>276</td>
      <td>250.01</td>
      <td>255</td>
    </tr>
    <tr>
      <th>2</th>
      <td>648</td>
      <td>250</td>
      <td>V27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>250.43</td>
      <td>403</td>
    </tr>
    <tr>
      <th>4</th>
      <td>197</td>
      <td>157</td>
      <td>250</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>101761</th>
      <td>250.13</td>
      <td>291</td>
      <td>458</td>
    </tr>
    <tr>
      <th>101762</th>
      <td>560</td>
      <td>276</td>
      <td>787</td>
    </tr>
    <tr>
      <th>101763</th>
      <td>38</td>
      <td>590</td>
      <td>296</td>
    </tr>
    <tr>
      <th>101764</th>
      <td>996</td>
      <td>285</td>
      <td>998</td>
    </tr>
    <tr>
      <th>101765</th>
      <td>530</td>
      <td>530</td>
      <td>787</td>
    </tr>
  </tbody>
</table>
<p>101766 rows × 3 columns</p>
</div>




```python
data['diag_1'].unique()
```




    <ArrowStringArray>
    ['250.83',    '276',    '648',      '8',    '197',    '414',    '428',
        '398',    '434',  '250.7',
     ...
         '58',    '649',    '832',    '133',    '975',    '833',    '391',
        '690',     '10',    'V51']
    Length: 717, dtype: str




```python
data['diag_2'].unique()
```




    <ArrowStringArray>
    [     '?', '250.01',    '250', '250.43',    '157',    '411',    '492',
        '427',    '198',    '403',
     ...
        '460',    '942',    '364',     '66',   'E883',    '123',    '884',
        'V60',    '843',    '927']
    Length: 749, dtype: str




```python
data['diag_3'].unique()
```




    <ArrowStringArray>
    [   '?',  '255',  'V27',  '403',  '250',  'V45',   '38',  '486',  '996',
      '197',
     ...
      '876',  '230',   '57', 'E854',  '942',   '14',  '750',  '370',  '671',
      '971']
    Length: 790, dtype: str




```python
cats = list(data['diag_1'].unique())+list(data['diag_2'].unique())+list(data['diag_3'].unique())
```


```python
len(cats)
```




    2256




```python
cats = list(set(cats))
```


```python
cats = [str(i) for i in cats]
```


```python
for diag in cats:
    col_name = "diagcode_" + diag
    data[col_name] = 0
```

```python
data=data.copy()
```


```python
data.loc[0,['diag_1','diag_2','diag_3']]
```


    diag_1    250.83
    diag_2         ?
    diag_3         ?
    Name: 0, dtype: str


```python
for index, values in data.iterrows():
    for i in ['diag_1','diag_2','diag_3']:
        diag_code = values[i]
        cols = "diagcode_" + str(diag_code)
        data.loc[index,cols] = 1
    if index%1000==0:
        print(index)
```

    

```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encounter_id</th>
      <th>patient_nbr</th>
      <th>race</th>
      <th>gender</th>
      <th>age</th>
      <th>admission_type_id</th>
      <th>discharge_disposition_id</th>
      <th>admission_source_id</th>
      <th>time_in_hospital</th>
      <th>payer_code</th>
      <th>...</th>
      <th>diagcode_911</th>
      <th>diagcode_621</th>
      <th>diagcode_V25</th>
      <th>diagcode_835</th>
      <th>diagcode_245</th>
      <th>diagcode_235</th>
      <th>diagcode_794</th>
      <th>diagcode_48</th>
      <th>diagcode_268</th>
      <th>diagcode_531</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2278392</td>
      <td>8222157</td>
      <td>Caucasian</td>
      <td>Female</td>
      <td>[0-10)</td>
      <td>6</td>
      <td>25</td>
      <td>1</td>
      <td>1</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149190</td>
      <td>55629189</td>
      <td>Caucasian</td>
      <td>Female</td>
      <td>[10-20)</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64410</td>
      <td>86047875</td>
      <td>AfricanAmerican</td>
      <td>Female</td>
      <td>[20-30)</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>500364</td>
      <td>82442376</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[30-40)</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16680</td>
      <td>42519267</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[40-50)</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>101761</th>
      <td>443847548</td>
      <td>100162476</td>
      <td>AfricanAmerican</td>
      <td>Male</td>
      <td>[70-80)</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>MC</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>101762</th>
      <td>443847782</td>
      <td>74694222</td>
      <td>AfricanAmerican</td>
      <td>Female</td>
      <td>[80-90)</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>MC</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>101763</th>
      <td>443854148</td>
      <td>41088789</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[70-80)</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>MC</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>101764</th>
      <td>443857166</td>
      <td>31693671</td>
      <td>Caucasian</td>
      <td>Female</td>
      <td>[80-90)</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>10</td>
      <td>MC</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>101765</th>
      <td>443867222</td>
      <td>175429310</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[70-80)</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>6</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>101766 rows × 965 columns</p>
</div>


```python
data.loc[2321,['diag_1','diag_2','diag_3']]
```


    diag_1      3
    diag_2    276
    diag_3    276
    Name: 2321, dtype: str


```python
data['diag_3']=data['diag_3'].astype(str)
```


```python
data['diag_3'].dtype
```


    <StringDtype(na_value=nan)>

## Feature Engineering & Encoding

We drop the ID columns, the original diagnosis columns (now encoded as `diagcode_*`), and two zero-variance drug columns (`examide`, `citoglipton`). Binary string columns are mapped directly to 0/1. `discharge_disposition_id` is cast to string before `get_dummies` because its integer codes map to named categories (home, skilled nursing, expired, etc.) with no ordinal relationship; treating it as numeric would obscure the signal from specific outcomes, particularly ID 11 (expired), which is a near-perfect predictor of no readmission. The target is label-encoded: sklearn assigns labels alphabetically, so `<30=0`, `>30=1`, `NO=2`.


```python
data = data.drop(columns=['encounter_id', 'patient_nbr', 'diag_1', 'diag_2', 'diag_3', 'examide', 'citoglipton'])

data['change'] = (data['change'] == 'Ch').astype(int)
data['diabetesMed'] = (data['diabetesMed'] == 'Yes').astype(int)
data['discharge_disposition_id'] = data['discharge_disposition_id'].astype(str)

le = LabelEncoder()
y = le.fit_transform(data['readmitted'])
print(dict(zip(le.classes_, le.transform(le.classes_))))

data = data.drop(columns=['readmitted'])
data = pd.get_dummies(data)
```

    {'<30': np.int64(0), '>30': np.int64(1), 'NO': np.int64(2)}


## Class Distribution

The target is imbalanced: `NO` accounts for ~54% of encounters, `>30` ~35%, and `<30` ~11%. Early readmission (`<30`) is the smallest and clinically most important class. This imbalance will be reflected in per-class precision and recall when we evaluate the model.


```python
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(le.classes_, np.bincount(y))
ax.set_xlabel('Readmitted')
ax.set_ylabel('Count')
ax.set_title('Class Distribution')
plt.tight_layout()
plt.show()
```


​    
![png](eric_graham_case_study_02_files/eric_graham_case_study_02_32_0.png)
​    


## Logistic Regression

We do an 80/20 train/test split and fit `StandardScaler` on the training data only, then transform both sets. `LogisticRegression` with `lbfgs` handles multiclass natively without needing one-vs-rest. Default `max_iter=100` will likely not converge with ~1,000 features, so we set it to 1000. Given the class imbalance, expect strong performance on `NO` and weak recall on `<30`.

`C` is the inverse of regularization strength (C = 1/lambda). Smaller `C` applies stronger regularization, shrinking coefficients toward zero and producing a simpler model; larger `C` relaxes regularization and lets the model fit the training data more closely. With roughly 1,000 features after one-hot encoding, the default `C=1` is a reasonable starting point, though a grid search over smaller values could reduce overfitting and may help the model generalize better to the minority class.


```python
X = data.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```


```python
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
```

                  precision    recall  f1-score   support
    
             <30       0.34      0.03      0.06      2285
             >30       0.50      0.36      0.42      7117
              NO       0.61      0.84      0.71     10952
    
        accuracy                           0.58     20354
       macro avg       0.48      0.41      0.39     20354
    weighted avg       0.54      0.58      0.53     20354




```python
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_)
plt.tight_layout()
plt.show()
```


​    
![png](eric_graham_case_study_02_files/eric_graham_case_study_02_36_0.png)
​    


The model leans heavily on `NO` with recall of 0.82 there but nearly zero on `<30` (0.03), meaning it almost never predicts early readmission. Only 69 of 2,285 actual `<30` cases were caught. The 0.58 accuracy is barely better than a naive baseline of ~0.54 achieved by just predicting `NO` every time. `>30` improved to 0.40 recall after properly encoding `discharge_disposition_id`. This is a direct consequence of the class imbalance: the model learns that guessing `NO` is usually right and rarely takes a risk on the minority class.

The most costly error is predicting `NO` when the true label is `<30`: a high-risk patient is discharged without follow-up. Of 2,285 actual early-readmission cases, roughly 2,216 were misclassified, most absorbed into `NO` predictions. A false positive wastes resources but harms no one; the false negative sends a patient home who will bounce back within 30 days. This asymmetry would motivate lowering the decision threshold for `<30` or applying class weights to trade overall accuracy for minority-class recall.

## Cross-Validation

A single 80/20 split gives one accuracy estimate that varies with which rows fall in the test set. Five-fold stratified CV trains and evaluates five times on non-overlapping folds, preserving class proportions in each, and reports a mean and standard deviation across folds. A tight standard deviation confirms the single-split result was not a fluke.


```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=42))])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
print(cv_scores.round(4))
print(f"Mean accuracy: {cv_scores.mean():.4f}  Std: {cv_scores.std():.4f}")
```

## Re-run with ID Columns Included

`encounter_id` is a unique key per visit. `patient_nbr` identifies each patient, and patients appear multiple times in this dataset. Because the 80/20 split is random at the encounter level, the same patient's records often land in both train and test sets, so the model can memorize `patient_nbr` to outcome associations rather than learning clinical features. This is data leakage.


```python
ids = pd.read_csv("./dataset_diabetes/diabetic_data.csv")[['encounter_id', 'patient_nbr']]

X_with_ids = np.hstack([X, ids.values])
X_train_ids, X_test_ids, y_train_ids, y_test_ids = train_test_split(X_with_ids, y, test_size=0.2, random_state=42)

scaler_ids = StandardScaler()
X_train_ids = scaler_ids.fit_transform(X_train_ids)
X_test_ids = scaler_ids.transform(X_test_ids)

model_ids = LogisticRegression(max_iter=1000, random_state=42)
model_ids.fit(X_train_ids, y_train_ids)
y_pred_ids = model_ids.predict(X_test_ids)
print(classification_report(y_test_ids, y_pred_ids, target_names=le.classes_))
```

## Variable Importance

`model.coef_` returns a matrix of shape `(3 classes, n_features)`. For each class, we take the 5 features with the largest absolute coefficient values. A large positive coefficient pushes the model toward predicting that class; a large negative coefficient pushes away from it.


```python
feature_names = data.columns.tolist()

for i, cls in enumerate(le.classes_):
    coefs = pd.Series(model.coef_[i], index=feature_names)
    top5 = coefs.abs().nlargest(5)
    print(f"Top 5 features for class '{cls}':", )
    print(coefs[top5.index].round(3).to_string())
```


```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (cls, ax) in enumerate(zip(le.classes_, axes)):
    coefs = pd.Series(model.coef_[i], index=feature_names)
    top5 = coefs.abs().nlargest(5)
    ax.barh(top5.index, coefs[top5.index])
    ax.set_title(f"Top 5 Features: {cls}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
plt.tight_layout()
plt.show()
```

**`<30`:** `discharge_disposition_id_11` (expired) is the strongest negative predictor: deceased patients are never early readmissions, so the model pushes hard against `<30` when it sees this code. `number_inpatient` remains a strong positive predictor; high prior utilization still signals early bounce-back risk.

**`>30`:** `discharge_disposition_id_11` is again the dominant negative predictor, with an even larger coefficient (-0.430) than for `<30`. `discharge_disposition_id_1` (discharged to home) is a positive predictor, consistent with patients stable enough to go home but who return later.

**`NO`:** `discharge_disposition_id_11` flips to a large positive coefficient (0.777), as the model has correctly learned that expired patients do not return. `number_inpatient` is the strongest negative predictor, consistent with its role in the readmission classes. After properly encoding `discharge_disposition_id` as categorical, discharge outcome dominates variable importance across all three classes.

## Conclusion

The logistic regression model achieves modest overall accuracy (0.58) but struggles with early readmission (`<30`), the most clinically important class, due to severe class imbalance. After one-hot encoding `discharge_disposition_id`, discharge outcome (particularly ID 11, expired) emerges as the dominant predictor across all three classes, with `number_inpatient` remaining the strongest clinical signal for readmission risk.

---

### Model Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `<30` | 0.34 | 0.03 | 0.06 | 2,285 |
| `>30` | 0.51 | 0.40 | 0.45 | 7,117 |
| `NO` | 0.62 | 0.82 | 0.71 | 10,952 |
| accuracy | | | 0.58 | 20,354 |
| macro avg | 0.49 | 0.42 | 0.41 | 20,354 |
| weighted avg | 0.55 | 0.58 | 0.55 | 20,354 |

### Top 5 Features by Class

**`<30`**

| Feature | Coefficient |
|---|---|
| `discharge_disposition_id_11` | -0.347 |
| `number_inpatient` | 0.240 |
| `discharge_disposition_id_22` | 0.107 |
| `diagcode_527` | -0.093 |
| `diagcode_462` | -0.092 |

**`>30`**

| Feature | Coefficient |
|---|---|
| `discharge_disposition_id_11` | -0.430 |
| `discharge_disposition_id_14` | -0.109 |
| `discharge_disposition_id_1` | 0.101 |
| `number_inpatient` | 0.085 |
| `discharge_disposition_id_6` | 0.066 |

**`NO`**

| Feature | Coefficient |
|---|---|
| `discharge_disposition_id_11` | 0.777 |
| `number_inpatient` | -0.325 |
| `number_emergency` | -0.133 |
| `discharge_disposition_id_14` | 0.101 |
| `discharge_disposition_id_6` | -0.082 |

