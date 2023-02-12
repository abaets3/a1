URL: https://github.com/abaets3/a1

TLDR; run python main.py after all packages installed.

To run this code you will need to install
numpy
pandas
scikit-learn
pytorch
matplotlib

You can use pip or conda or whatever you would like. I just did pip.

I have included the data in the data dir, so no need to download it

Once you have the correct packages installed, just run python main.py. 
This will generate ALL the charts and data used in my project write up. It will take a while to complete (10minutes+)
If you just want to execute a small portion, you can comment out the method calls at the very bottom of main.py.

a few notes:
-Nothing says the class name on the github so that it doesn't come up in search results
-I am planning to delete the repository after the semester ends
-Make sure you have a plots dir
-I did this on the linux subsystem on windows (WSL2)
-Should in theory work on any linux machine (I did not extensively test this)
-Does not require a GPU. I did not attach anything in pytorch to any GPUs.
-There is no seeding except for train/test splitting, therefore not all charts produced will match 100% to the project report
-There will be warnings about np.ravel, should still run fine.


TLDR; run python main.py after all packages installed.