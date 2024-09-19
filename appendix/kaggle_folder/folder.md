Let’s look at the structure of the files first of all. For any project that you are doing,
create a new folder. For this example, I am calling the project “project”.
The inside of the project folder should look something like the following.
.
├── input
│ ├── train.csv
│ └── test.csv
├── src
│ ├── create_folds.py
│ ├── train.py
│ ├── inference.py
│ ├── models.py
│ ├── config.py
│ └── model_dispatcher.py
├── models
│ ├── model_rf.bin
│ └── model_et.bin
├── notebooks
│ ├── exploration.ipynb
│ └── check_data.ipynb
├── README.md
└── LICENSE

Let’s see what these folders and file are about.
input/: This folder consists of all the input files and data for your machine learning
project. If you are working on NLP projects, you can keep your embeddings here.
If you are working on image projects, all images go to a subfolder inside this folder.
src/: We will keep all the python scripts associated with the project here. If I talk
about a python script, i.e. any _.py file, it is stored in the src folder.
models/: This folder keeps all the trained models.
notebooks/: All jupyter notebooks (i.e. any _.ipynb file) are stored in the notebooks
folder.
README.md: This is a markdown file where you can describe your project and
write instructions on how to train the model or to serve this in a production
environment.
LICENSE: This is a simple text file that consists of a license for the project, such as
MIT, Apache, etc. Going into details of the licenses is beyond the scope of this
book.
