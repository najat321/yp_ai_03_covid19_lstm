## a.	Project Title
Covid-19 New Cases Prediction
## b.	Project Description
### i.	What this project does?
This project is to to predict new cases (cases_new) in Malaysia using the past 30 days of number of cases by using using LSTM neural network.
### ii.	Any challenges that was faced and how I solved them?
- There were some challenges in the data cleaning process where there were a high number of missing values in the column of "cases_new","cluster_import", "cluster_religious","cluster_community","cluster_highRisk", "cluster_education", "cluster_detentionCentre","cluster_workplace" in the training dataset. I solved them by imputating the missing values using KNNImputer. 
- The data type of the "cases_new" column is an object datatype, so I solved it by converting it to an int64 datatype. 
### iii.	Some challenges / features you hope to implement?
I hope to implement features in the feature engineering process. Apart from the raw time series data, additional features can be created based on domain knowledge or patterns observed in the data. For example, you could derive lagged features, moving averages, or statistical measures such as mean, variance, or trend features. These additional features can provide valuable information to the LSTM model and potentially improve its performance.
## c.	How to install and run the project 
Here's a step-by-step guide on how to install and run this project:

1. Install Python: Ensure that Python is installed on your system. You can download the latest version of Python from the official Python website (https://www.python.org/) and follow the installation instructions specific to your operating system.

2. Clone the repository: Go to the GitHub repository where your .py file is located. Click on the "Code" button and select "Download ZIP" to download the project as a ZIP file. Extract the contents of the ZIP file to a location on your computer.

3. Set up a virtual environment (optional): It is recommended to set up a virtual environment to keep the project dependencies isolated. Open a terminal or command prompt, navigate to the project directory, and create a virtual environment by running the following command: python -m venv myenv
Then, activate the virtual environment:
If you're using Windows: myenv\Scripts\activate
If you're using Windows macOS/Linux: source myenv/bin/activate

4. Install dependencies: In the terminal or command prompt, navigate to the project directory (where the requirements.txt file is located). Install the project dependencies by running the following command: pip install -r requirements.txt
This will install all the necessary libraries and packages required by the project.

5. Run the .py file: Once the dependencies are installed, you can run the .py file from the command line. In the terminal or command prompt, navigate to the project directory and run the following command: python your_file.py
Now, you're done! The project should now run, and you should see the output or any other specified behavior defined in your .py file.

## d.	Output of this project
i. ![Alt Text](https://raw.githubusercontent.com/najat321/yp_ai_03_covid19_lstm/main/Matplotlib%20graph%20actual%20case%20vs%20predicted%20case.png?token=GHSAT0AAAAAACDTAPC3CUQPYRTSXM66KVJSZEAJU2Q)
## e.	Source of datasets : 
[https://github.com/MoH-Malaysia/covid19-public](https://github.com/MoH-Malaysia/covid19-public)

