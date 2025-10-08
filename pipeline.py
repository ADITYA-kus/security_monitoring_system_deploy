import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import json
from collections import defaultdict, Counter
import joblib
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

class ImprovedPredictiveMonitor:
    def __init__(self, location_clusters=15):
        self.model = None
        self.location_encoder = LabelEncoder()
        self.department_encoder = LabelEncoder()
        self.role_encoder = LabelEncoder()
        self.location_category_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_trained = False
        self.location_frequency_map = {}
        self.location_target_map = {}
        self.location_cluster_map = {}
        self.location_clusters = location_clusters
        self.location_hierarchy_map = {}
    # build frequency and target maps for location encoding    
    def _build_location_maps(self, features_data):
        location_counts = defaultdict(int)
        location_targets = defaultdict(list)

        # Collect location statistics
        for entity_id, entity_features in features_data.items():
            location_sequence = entity_features.get('sequence_features', {}).get('full_location_sequence', [])
            for loc in location_sequence:
                if loc and str(loc).strip() and str(loc) != 'UNKNOWN':
                    location_counts[str(loc)] += 1

            # For target encoding, track what locations follow each location
            if len(location_sequence) >= 2:
                for i in range(len(location_sequence) - 1):
                    current_loc = str(location_sequence[i])
                    next_loc = str(location_sequence[i + 1])
                    location_targets[current_loc].append(next_loc)

        self.location_frequency_map = location_counts

        # Calculate target encoding (most common next location)
        for loc, next_locs in location_targets.items():
            if next_locs:
                most_common = max(set(next_locs), key=next_locs.count)
                self.location_target_map[loc] = most_common
    # save trained model with all encoders and mappings            
    def save_model(self, filepath):
        if not self.is_trained:
             print(" No trained model to save")
             return False
    
        try:
          model_data = {
            'model': self.model,
            'location_encoder': self.location_encoder,
            'department_encoder': self.department_encoder,
            'role_encoder': self.role_encoder,
            'location_category_encoder': self.location_category_encoder,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'location_frequency_map': self.location_frequency_map,
            'location_target_map': self.location_target_map,
            'location_hierarchy_map': self.location_hierarchy_map,
            'location_clusters': self.location_clusters
        }
          joblib.dump(model_data, filepath)
          print(f"Model saved successfully to: {filepath}")
          print(f"Saved components: model, encoders, feature columns, location mappings")
          return True
        except Exception as e:
          print(f"Error saving model: {e}")
          return False
    # load trained model
    def load_model(self, filepath):
       try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.location_encoder = model_data['location_encoder']
            self.department_encoder = model_data['department_encoder']
            self.role_encoder = model_data['role_encoder']
            self.location_category_encoder = model_data['location_category_encoder']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            self.location_frequency_map = model_data['location_frequency_map']
            self.location_target_map = model_data['location_target_map']
            self.location_hierarchy_map = model_data['location_hierarchy_map']
            self.location_clusters = model_data.get('location_clusters', 15)
        
            print(f"Model loaded successfully from: {filepath}")
            print(f"Model type: {type(self.model).__name__}")
            print(f"Feature dimensions: {len(self.feature_columns)}")
            print(f"Available location categories: {list(self.location_encoder.classes_)}")
        
            return True
       except Exception as e:
           print(f" Error loading model: {e}")
           return False



    # create heirarchical location mapping to reduce target classes
    def _build_location_hierarchy(self, locations):
        hierarchy_map = {}
        
        for loc in locations:
            loc_str = str(loc).upper()
            
            # Map to broader categories to reduce target classes
            if 'LAB_' in loc_str:
                hierarchy_map[loc] = 'LAB'
            elif 'AUD_' in loc_str or 'AUDITORIUM' in loc_str:
                hierarchy_map[loc] = 'AUDITORIUM'
            elif 'HOSTEL' in loc_str or 'AP_HOSTEL' in loc_str:
                hierarchy_map[loc] = 'HOSTEL'
            elif 'LIB_' in loc_str or 'LIB_ENT' in loc_str:
                hierarchy_map[loc] = 'LIBRARY'
            elif 'CAF_' in loc_str or 'CAF_01' in loc_str:
                hierarchy_map[loc] = 'CAFETERIA'
            elif 'ADMIN' in loc_str:
                hierarchy_map[loc] = 'ADMIN'
            elif 'ROOM_' in loc_str or 'SEM_' in loc_str:
                hierarchy_map[loc] = 'CLASSROOM'
            elif 'GYM' in loc_str:
                hierarchy_map[loc] = 'GYM'
            else:
                # Keep frequent locations as-is, cluster others
                if self.location_frequency_map.get(loc, 0) > 2:  # Frequently visited
                    hierarchy_map[loc] = loc
                else:
                    hierarchy_map[loc] = 'OTHER'
                    
        return hierarchy_map
    # get categoray for a specific location
    def _get_location_category(self, location):
        loc_str = str(location).upper()
        if any(x in loc_str for x in ['LAB_', 'SEM_', 'CLASS', 'ROOM_']):
            return 'academic'
        elif any(x in loc_str for x in ['LIB_', 'STUDY', 'READING']):
            return 'library'
        elif any(x in loc_str for x in ['HOSTEL', 'DORM', 'RESIDENCE']):
            return 'residential'
        elif any(x in loc_str for x in ['CAF_', 'CAFETERIA', 'FOOD']):
            return 'dining'
        elif any(x in loc_str for x in ['GYM', 'SPORTS', 'RECREATION']):
            return 'recreational'
        elif any(x in loc_str for x in ['AUDITORIUM', 'HALL', 'AUD_']):
            return 'event'
        elif any(x in loc_str for x in ['ADMIN', 'OFFICE']):
            return 'administrative'
        else:
            return 'other'
    # fit all encoders onn the training data
    def _fit_encoders(self, X, y, features_data):
        # Collect all unique values for encoding
        all_departments = set()
        all_roles = set()
        all_location_categories = set()

        for entity_features in features_data.values():
            all_departments.add(str(entity_features.get('department', 'UNKNOWN')))
            all_roles.add(str(entity_features.get('role', 'unknown')))

        # Add location categories based on naming patterns
        location_categories = self._infer_location_categories(list(self.location_frequency_map.keys()))
        all_location_categories.update(location_categories)

        # Fit encoders
        self.department_encoder.fit(list(all_departments))
        self.role_encoder.fit(list(all_roles))
        self.location_encoder.fit(y)
        self.location_category_encoder.fit(list(all_location_categories))
    #infer location categories from location names
    def _infer_location_categories(self, locations):
        categories = set()
        for loc in locations:
            loc_str = str(loc).upper()
            if any(x in loc_str for x in ['LAB_', 'SEM_', 'CLASS', 'ROOM_']):
                categories.add('academic')
            elif any(x in loc_str for x in ['LIB_', 'STUDY', 'READING']):
                categories.add('library')
            elif any(x in loc_str for x in ['HOSTEL', 'DORM', 'RESIDENCE']):
                categories.add('residential')
            elif any(x in loc_str for x in ['CAF_', 'CAFETERIA', 'FOOD']):
                categories.add('dining')
            elif any(x in loc_str for x in ['GYM', 'SPORTS', 'RECREATION']):
                categories.add('recreational')
            elif any(x in loc_str for x in ['AUDITORIUM', 'HALL', 'AUD_']):
                categories.add('event')
            elif any(x in loc_str for x in ['ADMIN', 'OFFICE']):
                categories.add('administrative')
            else:
                categories.add('other')
        return categories

    def _get_time_period_from_hour(self, hour):
        """Convert hour to time period"""
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    def _safe_int_convert(self, value):
        """Safely convert value to integer, handling strings"""
        try:
            if isinstance(value, str):
                return int(value)
            elif isinstance(value, (int, float)):
                return int(value)
            else:
                return 12
        except (ValueError, TypeError):
            return 12

    def prepare_training_data(self, features_data, global_patterns):
        
       
        # First pass: Build location analysis
        self._build_location_maps(features_data)
        
        # Build location hierarchy to reduce target classes
        all_locations = list(self.location_frequency_map.keys())
        self.location_hierarchy_map = self._build_location_hierarchy(all_locations)
        

        X = []
        y = []
        entity_info = []

        for entity_id, entity_features in features_data.items():
            sequences = self._create_training_sequences(entity_features, entity_id)

            for sequence_data in sequences:
                features = self._extract_enhanced_features(sequence_data, global_patterns)
                
                # Use hierarchical location as target
                original_target = sequence_data['target_location']
                hierarchical_target = self.location_hierarchy_map.get(original_target, 'OTHER')
                
                if features and hierarchical_target and hierarchical_target != 'UNKNOWN':
                    X.append(features)
                    y.append(hierarchical_target)
                    entity_info.append({
                        'entity_id': entity_id,
                        'timestamp': sequence_data['timestamp'],
                        'context': sequence_data['context'],
                        'original_location': original_target,
                        'mapped_location': hierarchical_target
                    })

        if not X:
            return None, None, None

        # Fit encoders
        self._fit_encoders(X, y, features_data)

        # Encode features
        X_encoded_list = self._encode_features(X)
        y_encoded = self.location_encoder.transform(y)

        # Create feature array
        if X_encoded_list:
            self.feature_columns = list(X_encoded_list[0].keys())
            X_array = np.array([[d[col] for col in self.feature_columns] for d in X_encoded_list])
        else:
            X_array = np.array([])
            self.feature_columns = []

        return X_array, y_encoded, entity_info

    def _create_training_sequences(self, entity_features, entity_id):
      
        sequences = []

        location_sequence = entity_features.get('sequence_features', {}).get('full_location_sequence', [])
        temporal_features = entity_features.get('temporal_features', {})
        activity_features = entity_features.get('activity_features', {})

        # Clean location sequence
        clean_location_sequence = []
        for loc in location_sequence:
            if loc and str(loc).strip() and str(loc) != 'UNKNOWN' and str(loc) != 'None':
                clean_location_sequence.append(str(loc))

        if len(clean_location_sequence) < 2:
            return sequences

        # Calculate behavioral metrics
        total_movements = len(clean_location_sequence) - 1
        movement_regularity = temporal_features.get('activity_regularity', 0)
        preferred_hour = self._safe_int_convert(temporal_features.get('most_active_hour', 12))
        
        # Calculate hour consistency (how consistent are activity hours)
        peak_hours = temporal_features.get('peak_activity_hours', [])
        numeric_hours = [self._safe_int_convert(h) for h in peak_hours if self._safe_int_convert(h) >= 0]
        hour_consistency = len(set(numeric_hours)) / 24 if numeric_hours else 0

        # Create sequence examples with enhanced context
        for i in range(1, len(clean_location_sequence)):
            current_location = clean_location_sequence[i-1]
            next_location = clean_location_sequence[i]

            # Use temporal patterns for context
            current_hour = self._safe_int_convert(temporal_features.get('most_active_hour', 12))
            previous_locations = clean_location_sequence[max(0, i-4):i-1]

            sequence_data = {
                'current_location': current_location,
                'target_location': next_location,
                'timestamp': datetime.now(),
                'context': {
                    'department': str(entity_features.get('department', 'UNKNOWN')),
                    'role': str(entity_features.get('role', 'unknown')),
                    'current_hour': current_hour,
                    'time_period': self._get_time_period_from_hour(current_hour),
                    'previous_locations': previous_locations,
                    'location_frequency': entity_features.get('location_features', {}).get('frequent_locations', []),
                    'total_movements': total_movements,
                    'movement_regularity': movement_regularity,
                    'preferred_hour': preferred_hour,
                    'hour_consistency': hour_consistency,
                    'total_activities': activity_features.get('total_activities', 0),
                    'activity_density': activity_features.get('activity_density', 0),
                    'data_sources_used': len(activity_features.get('data_sources_used', [])),
                    'entity_id': entity_id
                }
            }

            sequences.append(sequence_data)

        return sequences

    def _extract_enhanced_features(self, sequence_data, global_patterns):
        
        features = {}

        # = ENHANCED TEMPORAL FEATURES =
        current_hour = sequence_data['context']['current_hour']
        features['current_hour'] = current_hour
        features['current_hour_sin'] = np.sin(2 * np.pi * current_hour / 24)
        features['current_hour_cos'] = np.cos(2 * np.pi * current_hour / 24)
        features['is_weekend'] = 1 if datetime.now().weekday() >= 5 else 0
        features['day_of_week'] = datetime.now().weekday()

        # Enhanced time period features
        time_period = sequence_data['context']['time_period']
        time_periods = ['morning', 'afternoon', 'evening', 'night']
        for period in time_periods:
            features[f'time_period_{period}'] = 1 if period == time_period else 0

        # ENHANCED ENTITY PROFILE
        features['department_raw'] = sequence_data['context']['department']
        features['role_raw'] = sequence_data['context']['role']

        #  ENHANCED LOCATION FEATURES 
        current_location = str(sequence_data['current_location'])
        hierarchical_location = self.location_hierarchy_map.get(current_location, 'OTHER')
        features['current_location_hierarchical'] = hierarchical_location
        
        # Location frequency with smoothing
        features['current_location_frequency'] = self.location_frequency_map.get(current_location, 1)
        features['location_frequency_log'] = np.log1p(features['current_location_frequency'])

        #  ENHANCED SEQUENCE FEATURES 
        previous_locations = [str(loc) for loc in sequence_data['context']['previous_locations']]
        features['previous_locations_count'] = len(previous_locations)
        
        # Sequence pattern features
        if previous_locations:
            # Recent location patterns
            recent_hierarchical = [self.location_hierarchy_map.get(loc, 'OTHER') for loc in previous_locations[-3:]]
            features['recent_location_variety'] = len(set(recent_hierarchical))
            
            # Transition patterns
            if len(previous_locations) >= 2:
                current_hierarchical = self.location_hierarchy_map.get(current_location, 'OTHER')
                prev_hierarchical = self.location_hierarchy_map.get(previous_locations[-1], 'OTHER')
                features['location_transition'] = hash(f"{prev_hierarchical}_{current_hierarchical}") % 1000
            else:
                features['location_transition'] = 0
                
        #  BEHAVIORAL PATTERN FEATURES 
        context = sequence_data['context']
        
        # Movement patterns
        features['total_movements'] = context.get('total_movements', 0)
        features['movement_regularity'] = context.get('movement_regularity', 0)
        
        # Temporal consistency
        features['preferred_hour'] = context.get('preferred_hour', 12)
        features['hour_consistency'] = context.get('hour_consistency', 0)
        
        # Department alignment
        department = context['department']
        dept_patterns = global_patterns.get('department_location_preferences', {}).get(department, [])
        features['department_alignment'] = len(dept_patterns)
        
        # Location category transition patterns
        current_category = self._get_location_category(current_location)
        features['current_location_category'] = current_category
        
        if previous_locations:
            prev_categories = [self._get_location_category(loc) for loc in previous_locations[-2:]]
            features['category_transition_count'] = len(set(prev_categories + [current_category]))
        else:
            features['category_transition_count'] = 1

        # ACTIVITY LEVEL FEATURES 
        features['total_activities'] = context.get('total_activities', 0)
        features['activity_density'] = context.get('activity_density', 0)
        features['data_sources_used'] = context.get('data_sources_used', 0)

        return features

    def _encode_features(self, X_raw):
        
        X_encoded = []

        for raw_features in X_raw:
            encoded_features = {}

            # Numerical features
            numerical_features = [
                'current_hour', 'current_hour_sin', 'current_hour_cos', 'is_weekend', 'day_of_week',
                'current_location_frequency', 'location_frequency_log', 'previous_locations_count',
                'recent_location_variety', 'location_transition', 'total_movements', 'movement_regularity',
                'preferred_hour', 'hour_consistency', 'department_alignment', 'category_transition_count',
                'total_activities', 'activity_density', 'data_sources_used'
            ]

            for feature in numerical_features:
                if feature in raw_features:
                    value = raw_features[feature]
                    if isinstance(value, (int, float, np.number)):
                        encoded_features[feature] = value
                    else:
                        try:
                            encoded_features[feature] = float(value)
                        except (ValueError, TypeError):
                            encoded_features[feature] = 0.0

            # Time period features
            for feature in ['time_period_morning', 'time_period_afternoon', 'time_period_evening', 'time_period_night']:
                if feature in raw_features:
                    encoded_features[feature] = raw_features[feature]

            # Categorical features with error handling
            categorical_mappings = [
                ('department_raw', 'department_encoded', self.department_encoder),
                ('role_raw', 'role_encoded', self.role_encoder),
                ('current_location_category', 'location_category_encoded', self.location_category_encoder),
                ('current_location_hierarchical', 'location_hierarchical_encoded', self.location_encoder)
            ]

            for raw_col, encoded_col, encoder in categorical_mappings:
                if raw_col in raw_features:
                    value = raw_features[raw_col]
                    try:
                        encoded_features[encoded_col] = encoder.transform([value])[0]
                    except ValueError:
                        encoded_features[encoded_col] = -1

            X_encoded.append(encoded_features)

        return X_encoded

    def train(self, features_data, global_patterns, test_size=0.2):
       

        X, y, entity_info = self.prepare_training_data(features_data, global_patterns)

        if X is None or X.size == 0:
            print(" Training failed - no data")
            return False

        

        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Try two model
        models = {
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        }

        best_score = 0
        best_model = None
        best_model_name = None

        for name, model in models.items():
            #print(f" Training {name}")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_model_name = name

        self.model = best_model
        self.is_trained = True

        
        # Feature importance analysis
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = list(zip(self.feature_columns, self.model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
        return True

    def _create_prediction_context(self, entity_features, current_time, global_patterns):
        
        recent_activities = entity_features.get('sequence_features', {}).get('recent_activities', [])

        if recent_activities:
            last_activity = recent_activities[0]
            current_location = last_activity.get('location', 'UNKNOWN')
            previous_locations = [act.get('location') for act in recent_activities[1:4] if act.get('location')]
        else:
            location_features = entity_features.get('location_features', {})
            current_location = location_features.get('most_visited_location', 'UNKNOWN')
            previous_locations = [loc for loc in location_features.get('frequent_locations', [])][:2]

        current_location = str(current_location) if current_location else 'UNKNOWN'
        previous_locations = [str(loc) for loc in previous_locations if loc and str(loc) != 'UNKNOWN']

        return {
            'current_location': current_location,
            'target_location': None,
            'timestamp': current_time,
            'context': {
                'department': str(entity_features.get('department', 'UNKNOWN')),
                'role': str(entity_features.get('role', 'unknown')),
                'current_hour': current_time.hour,
                'time_period': self._get_time_period_from_hour(current_time.hour),
                'previous_locations': previous_locations,
                'location_frequency': entity_features.get('location_features', {}).get('frequent_locations', []),
                'is_weekend': current_time.weekday() >= 5
            },
            'entity_features': entity_features  
        }
    # get top n prediction 
    def _get_top_predictions(self, probabilities, top_n=3):
        probabilities = np.asarray(probabilities)
        top_indices = np.argsort(probabilities)[-min(top_n, len(probabilities)):][::-1]

        if len(top_indices) == 0:
            return []

        top_locations = self.location_encoder.inverse_transform(top_indices)
        top_confidences = probabilities[top_indices]

        return [
            {'location': loc, 'confidence': conf}
            for loc, conf in zip(top_locations, top_confidences)
        ]

    def _generate_evidence(self, prediction_context, predicted_location, global_patterns):
       
        evidence = []
        context = prediction_context['context']
        current_hour = context['current_hour']
        time_period = context['time_period']
        department = context['department']
        role = context['role']
        
        # TEMPORAL EVIDENCE 
        evidence.append(f" Current time: {current_hour:02d}:00 ({time_period})")
        
        # Time-based patterns
        if 5 <= current_hour < 9:
            evidence.append("Early morning pattern: Typically academic/residential areas")
        elif 9 <= current_hour < 12:
            evidence.append("Morning pattern: High academic activity, classes/labs")
        elif 12 <= current_hour < 14:
            evidence.append("Lunch hours: Movement towards dining/recreational areas")
        elif 14 <= current_hour < 17:
            evidence.append("Afternoon pattern: Library/lab sessions, academic work")
        elif 17 <= current_hour < 20:
            evidence.append("Evening pattern: Recreational activities, social spaces")
        else:
            evidence.append("Night pattern: Residential areas, limited movement")
        
        # SEQUENCE & MOVEMENT EVIDENCE 
        previous_locations = context.get('previous_locations', [])
        if previous_locations:
            # Analyze movement patterns
            recent_sequence = previous_locations[-3:]  # Last 3 locations
            sequence_str = " → ".join(recent_sequence + [predicted_location])
            evidence.append(f"Recent movement pattern: {sequence_str}")
            
            # Movement type analysis
            if len(recent_sequence) >= 2:
                start_category = self._get_location_category(recent_sequence[0])
                end_category = self._get_location_category(predicted_location)
                
                if start_category != end_category:
                    evidence.append(f"Cross-category movement: {start_category} → {end_category}")
                else:
                    evidence.append(f"Same-category movement: Staying in {end_category} areas")
        
        # DEPARTMENT-SPECIFIC EVIDENCE 
        dept_patterns = global_patterns.get('department_location_preferences', {}).get(department, [])
        dept_locations = [loc for loc, freq in dept_patterns[:5]]
        
        if predicted_location in dept_locations:
            evidence.append(f" Department pattern: Common location for {department} students")
        elif dept_locations:
            evidence.append(f"Department context: {department} students often visit {', '.join(dept_locations[:2])}")
        
        #  roll based evidence
        if role == 'student':
            if time_period in ['morning', 'afternoon']:
                evidence.append("Student pattern: Likely in academic activities during day")
            else:
                evidence.append("Student pattern: Evening activities vary (study/social/residential)")
        elif role == 'faculty':
            evidence.append("Faculty pattern: Office hours, research areas, administrative duties")
        elif role == 'staff':
            evidence.append("Staff pattern: Administrative areas, consistent work locations")
        
        # LOCATION FREQUENCY EVIDENCE
        location_freq_list = context.get('location_frequency', [])
        predicted_visit_count = 0
        
        # Calculate visit frequency for predicted location
        for loc_info in location_freq_list:
            if isinstance(loc_info, dict) and loc_info.get('location') == predicted_location:
                predicted_visit_count = loc_info.get('frequency', 0)
                break
            elif loc_info == predicted_location:  # Handle string entries
                predicted_visit_count += 1
        
        if predicted_visit_count > 5:
            evidence.append(f"Frequently visited: {predicted_visit_count}+ previous visits (strong habit)")
        elif predicted_visit_count > 2:
            evidence.append(f"Regular location: {predicted_visit_count} previous visits (established pattern)")
        elif predicted_visit_count > 0:
            evidence.append(f"Occasional visit: {predicted_visit_count} previous visit(s)")
        else:
            evidence.append("New pattern: First predicted visit to this location")
        
        #  TIME CONSISTENCY EVIDENCE 
        entity_features = prediction_context.get('entity_features', {})
        temporal_features = entity_features.get('temporal_features', {})
        peak_hours = temporal_features.get('peak_activity_hours', [])
        
        if peak_hours and current_hour in [self._safe_int_convert(h) for h in peak_hours]:
            evidence.append("Peak activity hour: High probability based on historical patterns")
        
        # Check if this aligns with most active hour
        most_active_hour = temporal_features.get('most_active_hour')
        if most_active_hour and self._safe_int_convert(most_active_hour) == current_hour:
            evidence.append("Most active hour: Typically very active during this time")
        
        #  CONTEXTUAL INFERENCE 
        current_location_category = self._get_location_category(predicted_location)
        
        # Academic inference
        if current_location_category == 'academic':
            if 8 <= current_hour <= 16:
                evidence.append("Academic context: Normal class/lab hours")
            else:
                evidence.append("Late academic: Possible self-study/lab work")
        
        # Recreational inference
        elif current_location_category == 'recreational':
            if 16 <= current_hour <= 22:
                evidence.append("Recreational hours: Typical sports/social time")
            else:
                evidence.append("Off-hours recreation: Unusual but possible")
        
        # Residential inference
        elif current_location_category == 'residential':
            if current_hour <= 7 or current_hour >= 21:
                evidence.append("Residential night: Expected in hostel/dorm")
            else:
                evidence.append("Daytime residential: Possible break/rest")
        
        # Dining inference
        elif current_location_category == 'dining':
            if 7 <= current_hour <= 9:
                evidence.append("Breakfast hours: Expected at dining location")
            elif 12 <= current_hour <= 14:
                evidence.append("Lunch hours: Expected at dining location")
            elif 18 <= current_hour <= 20:
                evidence.append("Dinner hours: Expected at dining location")
            else:
                evidence.append("Off-meal dining: Snack/break time")
        
        #  WEEKEND VS WEEKDAY PATTERNS 
        is_weekend = context.get('is_weekend', False)
        if is_weekend:
            evidence.append("Weekend pattern: Different movement patterns expected")
            if current_location_category == 'academic':
                evidence.append("Weekend academics: Possible self-study/catch-up")
        else:
            evidence.append("Weekday pattern: Regular academic schedule")
        
        # CONFIDENCE-LEVEL EVIDENCE 
        confidence = prediction_context.get('confidence', 0)
        if confidence > 0.7:
            evidence.append("High confidence: Strong patterns support this prediction")
        elif confidence > 0.5:
            evidence.append("Moderate confidence: Reasonable inference from available data")
        else:
            evidence.append("Low confidence: Limited data for this context")
        
        return evidence

    def _display_prediction_results(self, prediction):
       print(f"\n PREDICTION RESULTS:")
       print(f"Location Category: {prediction['predicted_location']}")
       print(f"Confidence: {prediction['confidence']:.3f}")
    
       if prediction['specific_locations']:
        print(f"Likely Specific Locations: {', '.join(prediction['specific_locations'])}")
    
       print(f"\n EVIDENCE & REASONING:")
       for i, evidence in enumerate(prediction['evidence'][:8], 1):
        print(f"   {i}. {evidence}")
    
       if prediction['top_predictions'] and len(prediction['top_predictions']) > 1:
        print(f"\n ALTERNATIVE PREDICTIONS:")
        for alt in prediction['top_predictions'][1:]:
            print(f"  • {alt['location']} (confidence: {alt['confidence']:.3f})")
    # prediction with evidence        
    def predict_location(self, entity_features, global_patterns, current_time=None):
        if not self.is_trained:
            return None

        if current_time is None:
            current_time = datetime.now()

        prediction_context = self._create_prediction_context(entity_features, current_time, global_patterns)
        features = self._extract_enhanced_features(prediction_context, global_patterns)

        if not features:
            return None

        encoded_features_list = self._encode_features([features])
        if not encoded_features_list:
            return None

        encoded_features = encoded_features_list[0]
        
        try:
            features_array = np.array([[encoded_features.get(col, 0) for col in self.feature_columns]])
            prediction_encoded = self.model.predict(features_array)[0]
            prediction_proba = self.model.predict_proba(features_array)[0]
        except Exception as e:
            print(f" Prediction error: {e}")
            return None

        predicted_location = self.location_encoder.inverse_transform([prediction_encoded])[0]
        confidence = np.max(prediction_proba)
        
        # Get specific locations in this category
        specific_locations = [loc for loc, hier in self.location_hierarchy_map.items() 
                            if hier == predicted_location]
        
        # Sort by frequency for better recommendations
        specific_locations.sort(key=lambda x: self.location_frequency_map.get(x, 0), reverse=True)
        
        # Add confidence to context for evidence generation
        prediction_context['confidence'] = confidence
        
        return {
            'predicted_location': predicted_location,
            'specific_locations': specific_locations[:3],  
            'confidence': confidence,
            'top_predictions': self._get_top_predictions(prediction_proba, top_n=3),
            'context': prediction_context['context'],
            'evidence': self._generate_evidence(prediction_context, predicted_location, global_patterns)
        }
def run_improved_monitoring(features_file, save_model_path='trained_model.joblib'):
    """Run improved monitoring and save the trained model"""

    try:
        with open(features_file, 'r') as f:
            data = json.load(f)

        features_data = data['features']
        global_patterns = data['global_patterns']
        print(f" Loaded features for {len(features_data)} entities")
    except Exception as e:
        print(f" Error loading features: {e}")
        return

    # Initialize improved monitor
    monitor = ImprovedPredictiveMonitor(location_clusters=12)

    print("\n TRAINING IMPROVED MODEL")
    success = monitor.train(features_data, global_patterns)

    if success:
        # Save the trained model
        print(f"\n SAVING TRAINED MODEL...")
        save_success = monitor.save_model(save_model_path)
        
        if save_success:
            print(" Model saved successfully! Ready for production.")
        
        # Demo predictions
        sample_entities = list(features_data.keys())[:2]
        
        for entity_id in sample_entities:
            
            prediction = monitor.predict_location(
                features_data[entity_id],
                global_patterns
            )
            
            if prediction:
                monitor._display_prediction_results(prediction)
    
    return monitor if success else None


if __name__ == "__main__":
    features_file = 'predictive_features.json'
    run_improved_monitoring(features_file)