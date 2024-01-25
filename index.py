
from os import listdir
from pandas import read_csv
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np

# Load sequence for each subject, returns a list of numpy arrays
def load_dataset(prefix='./'):
    subjects = []
    directory = prefix
    for name in listdir(directory):
        filename = directory + name
        if filename.endswith('.csv'):
            df = read_csv(filename, header=None)
            # drop row number
            values = df.values[:, 1:]
            subjects.append(values)
    return subjects

# Scatter plot for a single subject's data with outliers marked for columns
def scatter_plot_with_outliers(subject, outliers):
    num_columns = subject.shape[1]

    # Create a scatter plot for each column
    for col in range(num_columns):
        plt.figure()
        plt.scatter(range(len(subject)), subject[:, col], label='Data Points', color='purple')

        # Mark outliers with red crosses
        if outliers is not None:
            outliers_mask = outliers == -1
            plt.scatter(
                np.where(outliers_mask)[0],  # x values
                subject[:, col][outliers_mask],  # y values
                label='Outliers',
                color='red',
                marker='x'
            )

        plt.xlabel('Data Point Index')
        plt.ylabel(f'Column {col+1}')
        plt.title(f'Scatter Plot for Column {col+1} with Outliers')
        plt.legend()
        plt.show()

# Scatter plot for the entire subject's data with outliers marked
def scatter_plot_subject_with_outliers(subject, outliers):
    # Create a scatter plot for the entire subject's data
    plt.figure()
    plt.scatter(range(len(subject)), subject, label='Data Points', color='purple')

    # Mark outliers with red crosses
    if outliers is not None:
        outliers_mask = outliers == -1
        flattened_outliers_mask = outliers_mask.flatten()[:len(subject)]  # Adjust the mask length
        plt.scatter(
            np.where(flattened_outliers_mask)[0],  # x values
            subject[flattened_outliers_mask],  # y values
            label='Outliers',
            color='red',
            marker='x'
        )

    plt.xlabel('Data Point Index')
    plt.ylabel('Values')
    plt.title('Scatter Plot for the Entire Subject with Outliers')
    plt.legend()
    plt.show()


# Outlier detection function using Isolation Forest
def detect_outliers_isolation_forest(subject):
    iso_forest = IsolationForest(contamination=0.1)
    outliers = iso_forest.fit_predict(subject)
    return outliers

# Load subjects
subjects = load_dataset()
print('Loaded %d subjects' % len(subjects))

# Choose a subject to plot
subject_to_plot = subjects[1]

# Detect outliers using Isolation Forest for the chosen subject
outliers = detect_outliers_isolation_forest(subject_to_plot.reshape(-1, 1))

# # Create scatter plots for each column with outliers marked
# scatter_plot_with_outliers(subject_to_plot, outliers)

# Create scatter plot for the entire subject with outliers marked
scatter_plot_subject_with_outliers(subject_to_plot.flatten(), outliers)
