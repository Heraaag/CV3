# Computer Vision III: Detection, Segmentation, and Tracking (IN2375)	
# Technical University Munich - WS 2026/27

## 1. Python Setup

If you are doing these exercises in Google Colab, you can skip this section and directly go to the notebook `0_exercise_intro.ipynb` in the folder `exercise_00`.

The exercises are tested with Python 3.12, the Python version of Google Colab. All required packages are listed in `requirements.txt`. We recommend a project-specific virtual environment, which you can create with either `pip` or [`uv`](https://docs.astral.sh/uv/). Both options use the same `requirements.txt`. The commands below are for Linux and macOS; on Windows, activate the environment with `.venv\Scripts\activate` instead. Installing `panopticapi` requires `git`.

### Option A: pip

Make sure Python 3.12 is installed, then run in the `cv3dst` folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter
```

### Option B: uv

uv downloads Python 3.12 automatically if it is not installed. Run in the `cv3dst` folder:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt jupyter
```

### PyTorch and GPUs

On Linux, the default `torch` package from PyPI comes with CUDA support. If you need a different CUDA version or a smaller CPU-only install, install `torch` and `torchvision` first as described on [pytorch.org](https://pytorch.org/get-started/locally/), e.g. for CPU only:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # with uv: uv pip install ...
```

and then install `requirements.txt` as above.

### Starting Jupyter

Activate the environment and start Jupyter:

```bash
source .venv/bin/activate
jupyter notebook
```

Jupyter uses the Python of the active environment, so always activate the environment before you work on the exercises. You can now go to the notebook `0_exercise_intro.ipynb` in the folder `exercise_00`.

## 2. Exercise Download

The exercises will be uploaded to the [course website](https://www.moodle.tum.de/course/view.php?id=122180). You from the links in the slides. Each time we start a new exercise you will have to unzip the exercise and copy it into the current directory as we are utilizing some shared folders.
### The directory layout for the exercises

    cv3dst
    ├── datasets            # The datasets will be stored here
    ├── exercise_00                 
    ├── exercise_01                 
    ├── exercise_02                     
    ├── exercise_03                    
    ├── exercise_04
    ├── exercise_05
    ├── models              # Where you will find all trained models, not for uploading
    ├── output              # Where you will find zipped exercises for uploading
    ├── README.md
    └── requirements.txt    # The python packages for pip, uv and Google Colab


## 3. Dataset Download

All datasets of the exercises are in `datasets.zip`, which you download from the same place as the exercises. Unzip it next to the exercise folders, so that the notebooks find it in the shared `datasets` folder:

    cv3dst
        └── datasets
            ├── MNIST           # exercise 00 (MNIST_png is created by the notebook)
            ├── PennFudanPed    # exercise 01, for academic and non-commercial use only
            ├── MOT16           # exercises 00, 02, 03
            ├── market          # exercise 02 (Market-1501)
            ├── reid_gnn        # exercise 03
            └── obj_seg         # exercises 04, 05


## 4. Exercise Submission
Your trained models will be automatically evaluated on a test set on our server. To this end, login or register for an account at:

https://cv3dst.cvai.cit.tum.de/

Note that only students who have registered for this class in TUM Online can register for an account. By using your matriculation number we send login data to your associated email address (probably the tum provided account if you didn't change it).

After you have worked through an exercise, execute the notebook cells that save and zip the exercise. The output can be found in the global `cv3dst/output` folder.

You can login to the above website and upload your zip submission for the current exercise. Your submission will be evaluated by our system.

You will receive an email notification with the results upon completion of the evaluation. To make the exercises more fun, you will be able to see a leaderboard of everyone's (anonymous) scores on the login part of the submission website.

Note that you can re-evaluate your models until the deadline of the current exercise. Whereas the email contains the result of the current evaluation, the entry in the leaderboard always represents the best score for the respective exercise.

## 5. Acknowledgments

We thank the **TU Munich Visual Computing and Artificial Intelligence Group** for creating this exercise framework.
