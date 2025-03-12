1. Overview
This project employs LSTM and PINN (Physics-Informed Neural Networks) to enhance the accuracy and interpretability of frozen soil prediction. The approach involves:
Temperature prediction using PIML.
Moisture content estimation with a physics-constrained LSTM model.
Freeze-thaw deformation analysis incorporating governing equations of frozen soil physics.
The dataset consists of multi-year monitoring data from the Menyuan region, Qinghai, China.
2. Files and Code Structure
The repository contains the following Python scripts:
PINN_move_9_10.py	Predicts freeze-thaw displacement using a physics-constrained LSTM model.
PINN101.9_37.6.py	Estimates soil temperature at different depths using PIML.
PINNTEXT_water_7.2.py	Models and predicts unfrozen water content in frozen soil using PINN.
3. Installation and Requirements
To run the scripts, install the following dependencies:
pip install numpy pandas torch scikit-learn matplotlib
Additionally, ensure you have Python 3.8+ installed.
4. Data Preparation
The scripts load soil data from CSV files. The dataset structure should include:
Timestamps (Date or time)
Soil temperature at different depths
Moisture content at different depths
Freeze-thaw displacement values
Other meteorological and hydrological features
Ensure the CSV file paths are correctly updated in the scripts before execution.
5. How to Run
Temperature Prediction
Run:
python PINN101.9_37.6.py
This script trains a physics-informed neural network to predict temperature variations in frozen soil.
Moisture Content Prediction
Run:
python PINNTEXT_water_7.2.py
This model estimates moisture content using governing equations related to soil water migration.
Frost Heave and Thaw Subsidence Prediction
Run:
python PINN_move_9_10.py
This script integrates physics-informed loss functions to predict freeze-thaw deformation.
6. Results and Output
The scripts generate:
Predictions vs. actual values saved as CSV files.
Figures comparing predicted and actual values.
Model loss and performance metrics displayed during training.
7. License
This project is open-source under the MIT License.
