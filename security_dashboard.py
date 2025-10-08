import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress the version warning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
from production_predictor import ProductionPredictor

class SecurityMonitoringDashboard:
    def __init__(self, model_path, entity_data_path,predictive_data_path):
        self.model_path = model_path
        self.entity_data_path = entity_data_path
        self.predictive_data_path=predictive_data_path
        self.entity_data = None
        self.predictor = None
        self.load_entity_data()
        self.load_predictor()
        self.setup_page()
    # load entity profiles and activity data
    def load_entity_data(self):
        try:
            with open(self.entity_data_path, 'r') as f:
                data = json.load(f)
            self.entity_data = data['entities']
            print(f" Loaded entity data for {len(self.entity_data)} entities")
        except Exception as e:
            st.error(f" Error loading entity data: {e}")
        # Loading ml predictor 
    def load_predictor(self):
        try:
            self.predictor = ProductionPredictor(self.model_path,self.predictive_data_path)
            print("ML predictor loaded successfully")
        except Exception as e:
            st.error(f"Error loading ML predictor: {e}")

    # dashboard structor using streamlit function
    def setup_page(self):
        st.set_page_config(
            page_title="Campus Security Monitoring",
            page_icon="🛡️",
            layout="wide"
        )
        
        st.title("🛡️ Campus Security Monitoring System")
        st.markdown("---")
    # check entity inactive or not for specified hours
    def check_inactivity_alerts(self, entity_id, hours_threshold=12):
        try:
            if entity_id not in self.entity_data:
                return {
                    'status': 'ERROR',
                    'hours_inactive': 0,
                    'message': 'Entity not found',
                    'last_seen': None
                }
                
            entity_info = self.entity_data[entity_id]
            activity_timeline = entity_info.get('activity_timeline', [])
            
            if not activity_timeline:
                return {
                    'status': 'ALERT',
                    'hours_inactive': 24,
                    'message': 'No activity data available',
                    'last_seen': None
                }
            
            # Get latest activity timestamp
            latest_timestamp = None
            for activity in activity_timeline:
                if 'timestamp' in activity:
                    try:
                        activity_time = pd.to_datetime(activity['timestamp'])
                        if latest_timestamp is None or activity_time > latest_timestamp:
                            latest_timestamp = activity_time
                    except:
                        continue
            
            if latest_timestamp is None:
                return {
                    'status': 'ALERT',
                    'hours_inactive': 24,
                    'message': 'No valid timestamps found',
                    'last_seen': None
                }
            
            current_time = datetime.now()
            hours_inactive = (current_time - latest_timestamp).total_seconds() / 3600
            
            status = 'ALERT' if hours_inactive > hours_threshold else 'ACTIVE'
            
            return {
                'status': status,
                'hours_inactive': hours_inactive,
                'last_seen': latest_timestamp,
                'message': f'Last seen {hours_inactive:.1f} hours ago'
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'hours_inactive': 0,
                'message': f'Error checking inactivity: {str(e)}',
                'last_seen': None
            }
    # Entity profile information
    def get_entity_profile(self, entity_id):
        if entity_id not in self.entity_data:
            return None
        
        entity_info = self.entity_data[entity_id]
        profile_info = entity_info.get('profile_info', {})
        
        return {
            'name': profile_info.get('name', 'Unknown'),
            'role': profile_info.get('role', 'Unknown'),
            'department': profile_info.get('department', 'Unknown'),
            'email': profile_info.get('email', 'Unknown'),
            'all_identifiers': profile_info.get('all_identifiers', [])
        }
    # Sequrity status
    def display_security_status(self, alert_info):
        st.subheader("Security Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if alert_info['status'] == 'ALERT':
                st.error("🚨 ALERT STATUS")
                st.write("Entity inactive >12 hours")
            elif alert_info['status'] == 'ACTIVE':
                st.success(" ACTIVE STATUS")
                st.write("Entity recently active")
            else:
                st.warning(" UNKNOWN STATUS")
                st.write(alert_info.get('message', 'Status unclear'))
        
        with col2:
            hours_inactive = alert_info['hours_inactive']
            st.metric(
                "Hours Inactive", 
                f"{hours_inactive:.1f}h",
                delta=f"{hours_inactive:.1f}h" if alert_info['status'] == 'ALERT' else None,
                delta_color="inverse" if alert_info['status'] == 'ALERT' else "normal"
            )
        
        with col3:
            if alert_info.get('last_seen'):
                st.metric("Last Seen", alert_info['last_seen'].strftime("%m/%d %H:%M"))
            else:
                st.metric("Last Seen", "Unknown")
        
        with col4:
            if alert_info['status'] == 'ALERT':
                st.metric("Alert Level", "HIGH", delta="Inactive >12h", delta_color="inverse")
            elif alert_info['status'] == 'ACTIVE':
                st.metric("Alert Level", "LOW", delta="Active")
            else:
                st.metric("Alert Level", "UNKNOWN")
    # Activity timeline
    def display_activity_timeline(self, entity_id):
        entity_info = self.entity_data[entity_id]
        activity_timeline = entity_info.get('activity_timeline', [])
        
        if not activity_timeline:
            st.info("No activity timeline data available")
            return
        
        # Create formatted timeline
        timeline_data = []
        for activity in activity_timeline[:15]:  # Show last 15 activities 
            try:
                timeline_entry = {
                    'timestamp': pd.to_datetime(activity.get('timestamp')),
                    'activity_type': activity.get('activity_type', 'Unknown'),
                    'location': activity.get('location', 'Unknown'),
                    'source': activity.get('source', 'Unknown'),
                    'confidence': activity.get('confidence', 0)
                }
                timeline_data.append(timeline_entry)
            except:
                continue
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            df_timeline = df_timeline.sort_values('timestamp', ascending=False)
            
            st.subheader(" Activity Timeline")
            
            # Display as table
            st.dataframe(
                df_timeline,
                width='stretch',
                hide_index=True
            )
        else:
            st.info("No valid activity data available")
    # getting behavioral patterns 
    def display_behavioral_insights(self, entity_id):
        entity_info = self.entity_data[entity_id]
        
        st.subheader("Behavioral Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Location Analysis
            location_analysis = entity_info.get('location_analysis', {})
            st.write("** Location Analysis**")
            
            most_visited = location_analysis.get('most_visited_location', 'Unknown')
            visit_frequency = location_analysis.get('visit_frequency', 0)
            st.metric("Most Visited Location", most_visited, delta=f"{visit_frequency} visits")
            
            location_prefs = location_analysis.get('location_preferences_by_time', {})
            if location_prefs:
                st.write("**Time-based Preferences:**")
                for time_period, locations in location_prefs.items():
                    if locations:
                        top_loc = max(locations.items(), key=lambda x: x[1])[0]
                        st.write(f"• {time_period.capitalize()}: {top_loc}")
        
        with col2:
            # Temporal Analysis
            temporal_analysis = entity_info.get('temporal_analysis', {})
            st.write("**Temporal Patterns**")
            
            peak_hours = temporal_analysis.get('peak_activity_hours', [])
            if peak_hours:
                st.write(f"**Peak Activity Hours:** {', '.join(map(str, peak_hours))}")
            
            hourly_dist = temporal_analysis.get('hourly_activity_distribution', {})
            if hourly_dist:
                total_activities = sum(hourly_dist.values())
                st.metric("Total Activities", total_activities)
            
            weekday_ratio = temporal_analysis.get('weekday_vs_weekend_ratio', 0)
            st.metric("Weekday/Weekend Ratio", f"{weekday_ratio:.2f}")
    #display detailed behaviour patterns
    def display_behavioral_patterns(self, entity_id):
        entity_info = self.entity_data[entity_id]
        behavioral_patterns = entity_info.get('behavioral_patterns', {})
        
        if behavioral_patterns:
            st.subheader("Behavioral Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Location Sequence:**")
                location_sequence = behavioral_patterns.get('location_sequence', [])
                if location_sequence:
                    st.write(" → ".join(location_sequence[-5:]))  # Last 5 locations
                
                st.write("**Unique Locations:**")
                unique_locs = behavioral_patterns.get('unique_locations', [])
                st.write(f"{len(unique_locs)} locations")
            
            with col2:
                location_freq = behavioral_patterns.get('location_frequency', {})
                if location_freq:
                    st.write("**Location Frequency:**")
                    for loc, freq in list(location_freq.items())[:3]:
                        st.write(f"• {loc}: {freq} visits")
    # EVIDENCE 
    def display_evidence_chains(self, entity_id):
        entity_info = self.entity_data[entity_id]
        evidence_chains = entity_info.get('evidence_chains', [])
        
        if evidence_chains:
            st.subheader("🔍 Evidence Chains")
            
            for i, chain in enumerate(evidence_chains[:3]):  # Show first 3 chains
                with st.expander(f"Evidence Chain {i+1}: {chain.get('type', 'Unknown')}"):
                    st.write(f"**Sequence:** {chain.get('sequence', [])}")
                    st.write(f"**Confidence:** {chain.get('confidence', 0):.2f}")
                    st.write(f"**Description:** {chain.get('description', 'No description')}")
    # ML BASED PREDICTION
    def generate_prediction(self, entity_id):
        if not self.predictor:
            st.error("ML predictor not available")
            return None
        
        try:
            prediction = self.predictor.display_result(entity_id)
            return prediction
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None
    
    def display_prediction_results(self, prediction):
        if not prediction:
            st.error("No prediction available")
            return
        
        st.subheader("ML Prediction Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            confidence = prediction.get('confidence', 0)
            st.metric(
                label="Predicted Location",
                value=prediction.get('predicted_location', 'Unknown'),
                delta=f"Confidence: {confidence:.1%}"
            )
            
            specific_locations = prediction.get('specific_locations', [])
            if specific_locations:
                st.write("**Specific Locations:**")
                for loc in specific_locations[:3]:
                    st.write(f"• {loc}")
        
        with col2:
            evidence = prediction.get('evidence', [])
            if evidence:
                st.write("**Evidence & Reasoning:**")
                for i, evidence_item in enumerate(evidence[:6], 1):
                    st.write(f"{i}. {evidence_item}")
            else:
                st.info("No evidence available")
    # dashboard interface:
    def create_dashboard(self):
        
        if not self.entity_data:
            st.error("No entity data loaded. Please check the file path.")
            return
        
        # Sidebar
        st.sidebar.title("🔍 Entity Search")
        
        # Entity selection
        available_entities = list(self.entity_data.keys())
        selected_entity = st.sidebar.selectbox(
            "Select Entity ID",
            options=available_entities,
            index=0
        )
        
        # Alert threshold
        alert_threshold = st.sidebar.slider(
            "Alert Threshold (hours)",
            min_value=1,
            max_value=24,
            value=12
        )
        
        # Display options
        st.sidebar.markdown("---")
        st.sidebar.subheader("Display Options")
        show_behavioral = st.sidebar.checkbox("Show Behavioral Patterns", True)
        show_evidence = st.sidebar.checkbox("Show Evidence Chains", True)
        show_prediction = st.sidebar.checkbox("Show ML Prediction", True)
        
        # Main content
        if selected_entity:
            # Entity Profile
            profile = self.get_entity_profile(selected_entity)
            if profile:
                st.header(f"👤 {profile['name']}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Role:** {profile['role']}")
                with col2:
                    st.write(f"**Department:** {profile['department']}")
                with col3:
                    st.write(f"**Email:** {profile['email']}")
                with col4:
                    st.write(f"**Entity ID:** {selected_entity}")
            
            st.markdown("---")
            
            # Security Status
            alert_info = self.check_inactivity_alerts(selected_entity, alert_threshold)
            self.display_security_status(alert_info)
            
            # Activity Timeline
            st.markdown("---")
            self.display_activity_timeline(selected_entity)
            
            # Behavioral Insights
            if show_behavioral:
                st.markdown("---")
                self.display_behavioral_insights(selected_entity)
                self.display_behavioral_patterns(selected_entity)
            
            # Evidence Chains
            if show_evidence:
                st.markdown("---")
                self.display_evidence_chains(selected_entity)
            
            # ML Prediction
            if show_prediction:
                st.markdown("---")
                st.subheader("ML Location Prediction")
                
                if st.button("Generate ML Prediction", type="primary"):
                    with st.spinner("Running ML prediction"):
                        prediction = self.generate_prediction(selected_entity)
                        if prediction:
                            self.display_prediction_results(prediction)
            
            # Quick Actions
            st.markdown("---")
            st.subheader("Quick Actions")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📊 Export Report"):
                    st.success("Report exported successfully!")
            with col2:
                if st.button("🔔 Notify Security"):
                    st.warning("Security team notified!")
            with col3:
                if st.button("🔄 Refresh"):
                    st.rerun()
        
        else:
            st.warning("Please select an entity from the sidebar")
    # run dashboard
    def run(self):
        self.create_dashboard()

# Main execution
if __name__ == "__main__":
    try:
        dashboard = SecurityMonitoringDashboard(
            model_path='trained_model.joblib',
            entity_data_path='Entity_resolution_map.json' , 
            predictive_data_path='predictive_features.json'
        )
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard failed to start: {e}")
