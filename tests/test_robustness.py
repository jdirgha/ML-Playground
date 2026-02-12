
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_handler import DataHandler

class TestDataHandlerRobustness(unittest.TestCase):
    def setUp(self):
        self.handler = DataHandler()
        
    def test_messy_dataset(self):
        """Test handling of ID columns, dates, and constants"""
        
        # Create a messy dataframe
        data = {
            'ID': [f'ID_{i}' for i in range(100)],  # High cardinality ID (should be dropped)
            'Date': pd.date_range(start='2023-01-01', periods=100),  # Date column (should be dropped)
            'Constant': [1] * 100,  # Constant column (should be dropped)
            'Feature1': np.random.rand(100),  # Good numeric feature
            'Feature2': np.random.choice(['A', 'B', 'C'], 100),  # Good categorical feature
            'Target': np.random.choice([0, 1], 100)  # Binary target
        }
        
        df = pd.DataFrame(data)
        self.handler.data = df
        
        # Manually set target (simulate user selection)
        self.handler.target_column = 'Target'
        self.handler.feature_columns = ['ID', 'Date', 'Constant', 'Feature1', 'Feature2']
        self.handler.task_type = 'Classification'
        
        # Run preprocessing
        print("Running preprocessing on messy data...")
        success = self.handler.preprocess_data()
        
        self.assertTrue(success, "Preprocessing failed")
        
        # Check if problematic columns were dropped
        train_cols = self.handler.X_train.columns
        
        print(f"Remaining columns: {list(train_cols)}")
        
        self.assertNotIn('ID', train_cols, "ID column was not dropped")
        self.assertNotIn('Date', train_cols, "Date column was not dropped")
        self.assertNotIn('Constant', train_cols, "Constant column was not dropped")
        self.assertIn('Feature1', train_cols, "Feature1 should be kept")
        self.assertIn('Feature2', train_cols, "Feature2 should be kept")
        
    def test_missing_values(self):
        """Test handling of missing values"""
        data = {
            'Numeric': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            'Categorical': ['A', 'B', np.nan, 'A', 'B', 'A', 'B', 'A', 'B', 'A'],
            'Target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        self.handler.data = df
        self.handler.target_column = 'Target'
        self.handler.feature_columns = ['Numeric', 'Categorical']
        self.handler.task_type = 'Classification'
        
        success = self.handler.preprocess_data()
        self.assertTrue(success)
        
        # Check no NaN in training data
        self.assertFalse(self.handler.X_train.isnull().any().any(), "Training data still has NaNs")

if __name__ == '__main__':
    unittest.main()
