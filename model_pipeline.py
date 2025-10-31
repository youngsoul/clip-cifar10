import time
from pathlib import Path
from typing import Literal, List

import numpy as np
import pandas as pd
from embetter.grab import ColumnGrabber
from embetter.multi import ClipEncoder
from embetter.vision import ImageLoader
from joblib import dump, load
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

SOLVER = 'liblinear'

# CIFAR10
# target_dirs = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# root_dir = str(Path(__file__).parent.resolve()) + "/CIFAR-10-images"


target_dirs = ['car', 'motorcycle', 'truck', 'frog']
root_dir = str(Path(__file__).parent.resolve()) + "/drone_images"

# show _all_ columns when you print a DataFrame:
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)       # auto‐wrap to your terminal width
pd.set_option('display.max_colwidth', None)  # don't truncate column contents

def generate_image_embeddings(embeddings_file: str = "embeddings.joblib", dir_name: Literal['train','test'] = 'train'):
    """
    Generates image embeddings for the specified directory and saves them to a specified file.
    It checks if the embeddings file already exists. If not, it processes images in the given
    directory ('train' or 'test'), encodes them into embeddings using a pipeline, and saves
    the embeddings along with their corresponding targets. The function is designed to support
    efficient reuse of precomputed embeddings.

    :param embeddings_file: Path to the file where image embeddings will be saved or loaded from.
                     If the file already exists, the function skips the embedding generation step.
    :type embeddings_file: str
    :param dir_name: Name of the directory ('train' or 'test') containing the images to be
                     processed for embedding generation.
    :type dir_name: Literal['train', 'test']
    :return: None
    """
    print(f"Generating image embeddings for dir: {dir_name}")

    if not Path(embeddings_file).exists():
        training_files_df = create_filepaths_df(dir_name=dir_name, dirs=target_dirs)

        # create pipeline to read the filepath column, load the image, and encode the image
        image_embedding_pipeline = make_pipeline(
           ColumnGrabber("filepath"),
          ImageLoader(convert="RGB"),
          ClipEncoder(),
        )

        # convert the filepaths to embeddings
        X = image_embedding_pipeline.fit_transform(training_files_df)
        y = training_files_df['target']
        print(X.shape)
        print(y.shape)

        # Save both together
        dump((X, y), embeddings_file)
    else:
        print("Embeddings file already exists. Skipping.")

def get_image_embeddings(embeddings_file: str = "embeddings.joblib"):
    X, y = load(embeddings_file)
    return X, y


def create_filepaths_df(dir_name: Literal['train','test'], dirs: List = target_dirs) -> pd.DataFrame:
    """
    Creates a pandas DataFrame containing file paths and their respective target
    categories from the specified directories under the given parent directory.

    :param dir_name: The root directory name, either 'train' or 'test',
        indicating whether data is for training or testing.
    :type dir_name: Literal['train', 'test']
    :param dirs: List of subdirectories under the root directory to be scanned for files.
    :type dirs: List
    :return: A DataFrame containing columns 'filepath' with the absolute file paths
        and 'target' with the corresponding categories based on the subdirectory names.
    :rtype: pandas.DataFrame
    """
    data = []
    for dir in dirs:
        for file in Path(f'{root_dir}/{dir_name}/{dir}').glob('*.jpg'):
            row_data = {
                'filepath': file,
                'target': dir
            }
            data.append(row_data)
    files_df = pd.DataFrame(data, columns=["filepath", "target"])
    return files_df


def train_grid_search() -> LogisticRegression:
    """
    Trains a Logistic Regression model using Randomized Search with Stratified Cross-Validation
    to identify the best set of hyperparameters. The function leverages pre-generated image
    embeddings as input features and targets (X, y) for model training.

    Prior to training, this function defines a distribution of hyperparameters. RandomizedSearchCV
    is employed to explore the hyperparameter space efficiently. The best model is selected based on
    the highest cross-validation accuracy, and the corresponding hyperparameters are logged.

    The StratifiedKFold cross-validation ensures the class distribution is preserved in both training
    and validation folds. Finally, the model, optimized with the best hyperparameters, is refit on the
    entire dataset before being returned.

    :returns: A LogisticRegression instance optimized with the best hyperparameter configuration
              as determined by RandomizedSearchCV.
    :rtype: LogisticRegression
    """
    print("Grid Search Training the model")

    generate_image_embeddings()
    X, y = get_image_embeddings()

    # 4. Define hyperparameter distributions
    param_distributions = {
        "C": loguniform(1e-4, 1e4),
        "penalty": ["l1", "l2", "elasticnet", "none"],
        "solver": ["saga"],  # saga supports all penalties
        "l1_ratio": np.linspace(0, 1, 5),
        "max_iter": [500, 1000, 2000]
    }

    # create a baseline, default model
    model = LogisticRegression(n_jobs=-1)

    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 6. Set up RandomizedSearchCV with stratified CV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="accuracy",
        cv=strat_cv,  # <-- use StratifiedKFold here
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # 7. Run the search
    random_search.fit(X, y)

    # 8. Examine the best results
    print("Best CV accuracy: {:.3f}".format(random_search.best_score_))
    print("Best hyperparameters:")
    for param in sorted(param_distributions):
        print(f"  {param}: {random_search.best_params_[param]}")

    # Retrieve the best pipeline (scaler + LogisticRegression) found:
    best_model = random_search.best_estimator_

    # the best model is re-fit on all of the data so you do not have to do that below
    # train the model on all of the data
    # best_model = LogisticRegression(solver='liblinear', max_iter=1_000)
    # best_model.fit(X, y)

    return best_model


def train(model_joblib_name: str = "model.joblib") -> LogisticRegression:
    """
    Trains a logistic regression model if no pretrained model is found. This function is designed
    to handle both the training process and loading of an existing model. It trains the model based
    on image embeddings and evaluates its performance using cross-validation. The trained model
    is saved as a file for future use.

    :return: The trained or preloaded logistic regression model
    :rtype: LogisticRegression
    """
    if not Path(model_joblib_name).exists():
        print("Training the model")
        generate_image_embeddings()
        X, y = get_image_embeddings()

        print(X.shape)
        print(y.shape)

        # create a baseline, default model
        model = LogisticRegression(solver=SOLVER, max_iter=1000, n_jobs=-1)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = cross_val_score(model, X, y, cv=cv, verbose=1, scoring='accuracy')
        print(scores)
        print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

        # train the model on all of the data
        print("Training the model on all of the data")
        model = LogisticRegression(solver=SOLVER, max_iter=1000, n_jobs=-1)
        model.fit(X, y)

        dump(model, model_joblib_name)
    else:
        print("Model already exists. Skipping.")
        model = load('model.joblib')

    return model

def validate(model: LogisticRegression, validate_embeddings_file: str = "valid_embeddings.joblib"):
    """
    Validates a logistic regression model using pre-generated image embeddings. The
    function evaluates the performance of the model by calculating accuracy and
    generating a confusion matrix, which provides a detailed comparison of the
    actual and predicted values.

    :param model: A logistic regression model used for predictions, which should
        already be trained.
    :type model: LogisticRegression
    :return: None
    """
    print("Validating the model")
    generate_image_embeddings(embeddings_file=validate_embeddings_file, dir_name='test')
    X, y = get_image_embeddings(embeddings_file=validate_embeddings_file)


    y_pred = model.predict(X)
    print(accuracy_score(y, y_pred))

    # Confusion matrix
    print("\nConfusion Matrix:")
    confusion_matrix = pd.crosstab(
        y,
        y_pred,
        rownames=['Actual'],
        colnames=['Predicted'],
        dropna=False
    )
    print(confusion_matrix)



if __name__ == "__main__":
    total_s = time.time()
    s = time.time()
    model = train()
    e = time.time()
    print(f"Training time: {e-s}")

    s = time.time()
    validate(model)
    e = time.time()
    print(f"Validation time: {e-s}")

    total_e = time.time()
    print(f"Total time: {total_e-total_s}")