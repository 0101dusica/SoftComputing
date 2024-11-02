 # K1-Segmentation

## Overview
    The goal of this project is to develop a solution that performs segmentation on a dataset of images, counting specific objects (toad, boo, and bobomb) in each image. The solution should calculate the total counts for these objects and minimize the Mean Absolute Error (MAE) when compared to the true counts provided in the object_count.csv file.

## Requirements
    - Python 3.10.x

## How to Run

1. **Navigate to the Script Folder**:  
   Open a terminal or command prompt, then navigate to the `k1-segmentation` directory.

   ```bash
   cd k1-segmentation

2. **Run the Script:**:  
    Execute the following command:

    ```bash
    python main.py data/

3. **Expected Output:**:  
    After running, the console will display the Mean Absolute Error (example result shown below):

    ```bash
    Mean Absolute Error: 1.0

    
    A result.txt file will also be generated with detailed information.

## Example result.txt Output

    The file will contain details such as predictions versus true values for each image, along with the Mean Absolute Error and elapsed time.

    Picture              Prediction      Truth
    ---------------------------------------------
    picture_1.png        9               10
    picture_10.png       6               6
    picture_2.png        20              23
    picture_3.png        15              16
    picture_4.png        9               9
    picture_5.png        4               3
    picture_6.png        9               10
    picture_7.png        3               3
    picture_8.png        14              13
    picture_9.png        15              17
    ---------------------------------------------

    Mean Absolute Error: 1.0
    Elapsed time: 2.19 seconds

## Notes
    The 1.0 in the example above is just a placeholder result and will vary based on the actual data.
    Make sure to provide the correct path to your data directory when running the script.