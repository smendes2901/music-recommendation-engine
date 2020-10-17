The following steps need to be carried out to implement the code for the music recommendation system

1. Download/Copy the MSc_Research_Project_Code.zip onto a local path and unzip it.
2. Download the fma_metadata.zip and fma_small.zip files from https://github.com/mdeff/fma
3. Create a folder called Dataset in MSc_Research_Project_Code.
4. Unzip fma_metadata.zip and fma_small.zip files into the Dataset folder in MSc_Research_Project_Code.
5. Check the requirements.txt file for the packages and libraries version requirements.
6. Run the eda.ipynb and preprocessing.ipynb files in Jupyter Notebook.
7. The preprocessing.ipynb file will create two new folders of test and train in MSc_Research_Project_Code. 
8. 3 npy files will be generated in each of the folders. The 3 npy files are classes.npy, features.npy and names.npy.
9. Create a google account and upload the updated MSc_Research_Project_Code folder onto Google Drive. Get a Google Colab Pro account for running the code.
10. Run the train.ipynb file on Google Colab Pro. A checkpoint folder will be created by the system with subfolders for the 3 models. 
11. Each of these folders will contain class weight files in h5 format corresponding to the epochs number. 29 h5 files will be generated.
12. You will be prompted to input the epoch at which the model performed the best(High accuracy and low loss which can be viewed in the console). The evaluation metrics will be printed for it on the console.
13. You will be prompted to input 'Y' or 'N' if you want that model to be saved and used further. On inputting Y the model will be saved in a system created best_models folder.
If you input 'N' then you will be prompted to input another epoch and then select 'Y' or 'N' for saving it.
14. Steps 10 and 11 will be carried out 3 times where the user will be prompted for the 3 models.
15. Run the recommender.ipynb file in Google Colab Pro. The user will be prompted to input an anchor song from the above-mentioned songs in the console and also input the number of songs to be recommended. The recommended songs will be printed on the console.