## Feature Selection Methods
Distance Correlation Based methods
Uses a first pass of distance correlation maximization to select
a proposed feature set..


Then uses partial distance correlation to remove  features that don't
individually contribute

python 3.8

```python
from FeatureSelectors.Distancecorrelation import Selector
x =  np.random.normal(1, size=(100, 10))
y =  np.random.choice([0,1], size=(100)).astype(np.float32)
results = Selector().fit_transform(x, y)
print(results)

```

Scaler Method that ignores Null and Nan Inputs, the learn scaling
for the cases when imputation happens after scaling.


```python
from FeatureSelectors.Distancecorrelation import Scaler
x = np.random.choice([0,.5, -5., 10,  1, None, np.nan], size=(100,10))
s = Scaler().fit(x)
results = s.inverse_transform(s.transform(x, sample_size=10))


####  Start the tracking server
```python
mlflow server --backend-store-uri sqlite:///models.db --default-artifact-root ./mlruns

```
before runnning the first time, create the database using sqlite3
