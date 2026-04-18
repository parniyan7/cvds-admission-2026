import csv
import numpy as np
import matplotlib.pyplot as plt


def plot_data(csv_file_path: str):
    """
    Plots a Precision-Recall curve from a CSV file containing precision and recall values.
    
    Original issues in the admission notebook:
    - csv.reader reads everything as strings (object dtype).
    - np.stack(results, dtype=float) failed with a TypeError because the data was still strings.
    - Matplotlib could not plot the curve correctly.
    
    How I fixed it:
    - After reading the rows, I convert the entire list to a numpy array using np.array(..., dtype=float).
    - Added clear labels, title, grid, and axis limits for better readability.
    - Kept the plot simple and exactly as expected by the admission test.
    """
    # Load data from CSV file
    results = []
    with open(csv_file_path, mode='r') as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=',')
        next(csv_reader)  # Skip the header row ("precision", "recall")
        for row in csv_reader:
            results.append(row)
    
    # Convert to numpy array with float values (this was the main bug fix)
    results = np.array(results, dtype=float)
    
    # Plot the precision-recall curve
    plt.plot(results[:, 1], results[:, 0])   # x = recall, y = precision
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()


# ========================
# Test 
# ========================
if __name__ == "__main__":
    # Create the test CSV file
    with open("data_file.csv", "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["precision", "recall"])
        w.writerows([
            [0.013, 0.951],
            [0.376, 0.851],
            [0.441, 0.839],
            [0.570, 0.758],
            [0.635, 0.674],
            [0.721, 0.604],
            [0.837, 0.531],
            [0.860, 0.453],
            [0.962, 0.348],
            [0.982, 0.273],
            [1.0, 0.0]
        ])
    
    # Run the function
    plot_data('data_file.csv')