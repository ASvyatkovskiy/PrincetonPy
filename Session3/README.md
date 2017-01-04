# Getting started (before the day of the workshop)

To be able to follow the workshop exercises, you are going to need a laptop with Anaconda and Firefox browser installed. 
Following instruction are geared for Mac or linux users.

## Download and install Anaconda

Please go to the following website: https://www.continuum.io/downloads
download and install *the latest* Anaconda version for Python 2.7 for your operating system. 

Note: we are going to need Anaconda 4.1.x or later

After that, type:

```bash
conda --help
```
and read the manual.
Once Anaconda is ready, download the following requirements file: https://github.com/ASvyatkovskiy/PrincetonPy/blob/master/Session3/requirements.txt
and proceed with setting up the environment:

```bash
conda create --name ScrapingWorkshop --file requirements.txt
source activate ScrapingWorkshop
```

## Install drivers required by Firefox

For Mac users, install geckodrivers using `brew`:

```bash
#brew update
brew install geckodriver
```

# Check-out the git repository with the exercise (on the day of the workshop)

```bash
https://github.com/ASvyatkovskiy/PrincetonPy
```

If you do not have git or you do not wish to install it, just download the repository as zip, and unpack it:

```bash
wget https://github.com/ASvyatkovskiy/PrincetonPy/archive/master.zip
#For Mac users:
#curl -O https://github.com/ASvyatkovskiy/PrincetonPy/archive/master.zip
unzip master.zip
```

## Start the interactive notebook

Change to the the repository folder, and start interactive jupyter (ipython) notebook:
```bash
cd PrincetonPy/Session3
jupyter notebook
```
