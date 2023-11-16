# Example Prediction Model

1. Explain what the model is and a brief description of what it's doing/how is works

2. Provide references for the model

3. Provide a list of strengths and weaknesses, and assumptions of the method

4. Provide any other information that would be helpful to users of the model

## Loading a pre-trained predictor

This is done in `evaluate.py` and `assess_forecasts.py`.

```python
from models import ExamplePredictor
predictor = ExamplePredictor(N=6, tau=48)
```

## Running files inside the module
The following files are provided inside the module, please run these from the root directory as a module, for example:
```
python -m models.example.example_test.py
```


## Directory Structure

Outline the structure of the directory and the files you have provided. E.g.,

- `example_model.py` - implementation of example model
- `example_train.py` - script for training example model
- `example_test.py` - script for testing example model in the following way ...
- `resources` - contains data files for pre-trained version of model: example1, example2, etc.
- ...

## Any other notes

E.g. preliminary results, suggestions for further work and improvements, points worth mentioning in the journal paper.