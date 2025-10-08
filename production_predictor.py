import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
from pipeline import ImprovedPredictiveMonitor

class ProductionPredictor:
    def __init__(self, model_path, data_path=None):
        self.model_path = model_path
        self.data_path = data_path
        self.monitor = None
        self.features_data = None
        self.global_patterns = None
        self.load_model()
        if data_path:
            self.load_data()
    # load trained model
    def load_model(self):
        try:
            self.monitor = ImprovedPredictiveMonitor()
            
            # Load the saved model data
            model_data = joblib.load(self.model_path)
            
            self.monitor.model = model_data['model']
            self.monitor.location_encoder = model_data['location_encoder']
            self.monitor.department_encoder = model_data['department_encoder']
            self.monitor.role_encoder = model_data['role_encoder']
            self.monitor.location_category_encoder = model_data['location_category_encoder']
            self.monitor.feature_columns = model_data['feature_columns']
            self.monitor.is_trained = model_data['is_trained']
            self.monitor.location_frequency_map = model_data['location_frequency_map']
            self.monitor.location_target_map = model_data['location_target_map']
            self.monitor.location_hierarchy_map = model_data['location_hierarchy_map']            
        except Exception as e:
            print(f"Error loading production model: {e}")
            raise
    
    def load_data(self):
        """Load the predictive_features data"""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            self.features_data = data['features']
            self.global_patterns = data['global_patterns']
            print(f"Loaded data for {len(self.features_data)} entities")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    def predict_location_api(self, entity_id, current_time=None):
        if not self.features_data or entity_id not in self.features_data:
            return None
    
        if current_time is None:
          current_time = datetime.now()
    
        entity_data = self.features_data[entity_id]
        prediction = self.monitor.predict_location(entity_data, self.global_patterns, current_time)
    
        return prediction



    def display_result(self, entity_id, current_time=None):
        if not self.features_data:
            raise ValueError("No features data loaded")
        
        if entity_id not in self.features_data:
            raise ValueError(f"Entity {entity_id} not found in features data")
        
        if current_time is None:
            current_time = datetime.now()
        
        # Use the already loaded monitor
        entity_data = self.features_data[entity_id]
        prediction = self.monitor.predict_location(entity_data, self.global_patterns, current_time)
        
        if prediction:
            print(f"\n{'='*6}")
            print(f"PREDICTION FOR: {entity_id}")
            print(f"{'='*60}")
            self.monitor._display_prediction_results(prediction)  
        else:
            print(f"No prediction returned for {entity_id}")
        
        return prediction
    
    def get_available_entities(self):
        if not self.features_data:
            return []
        return list(self.features_data.keys())
    
   
    

pre = ProductionPredictor('trained_model.joblib', 'predictive_features.json')

#prediction1 = pre.display_result('E106121')   