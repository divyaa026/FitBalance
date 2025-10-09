"""
API Integration for Fitbit/Oura Health Data
Real-time health metrics collection for protein optimization
"""

import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta, date
import json
import os
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class FitbitCredentials:
    """Fitbit API credentials"""
    access_token: str
    refresh_token: str
    client_id: str
    client_secret: str
    expires_at: datetime

@dataclass
class OuraCredentials:
    """Oura API credentials"""
    access_token: str
    refresh_token: str
    client_id: str
    client_secret: str

@dataclass
class HealthDataPoint:
    """Unified health data point from any source"""
    date: date
    source: str  # 'fitbit', 'oura', 'manual'
    sleep_duration: Optional[float] = None
    sleep_quality: Optional[float] = None
    hrv: Optional[float] = None
    resting_heart_rate: Optional[float] = None
    activity_score: Optional[float] = None
    steps: Optional[int] = None
    calories_burned: Optional[float] = None
    stress_level: Optional[float] = None
    recovery_score: Optional[float] = None
    raw_data: Optional[Dict] = None

class FitbitAPI:
    """Fitbit API integration for health metrics"""
    
    def __init__(self, credentials: FitbitCredentials = None):
        self.credentials = credentials
        self.base_url = "https://api.fitbit.com/1"
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated request to Fitbit API"""
        if not self.credentials:
            raise ValueError("Fitbit credentials not provided")
        
        session = await self._get_session()
        headers = {
            'Authorization': f'Bearer {self.credentials.access_token}',
            'Accept': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 401:
                    # Token expired, refresh
                    await self._refresh_token()
                    headers['Authorization'] = f'Bearer {self.credentials.access_token}'
                    async with session.get(url, headers=headers, params=params) as retry_response:
                        return await retry_response.json()
                
                response.raise_for_status()
                return await response.json()
                
        except Exception as e:
            logger.error(f"Fitbit API request failed: {e}")
            return {}
    
    async def _refresh_token(self):
        """Refresh Fitbit access token"""
        try:
            session = await self._get_session()
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': self.credentials.refresh_token,
                'client_id': self.credentials.client_id,
            }
            
            auth = aiohttp.BasicAuth(self.credentials.client_id, self.credentials.client_secret)
            
            async with session.post('https://api.fitbit.com/oauth2/token', 
                                  data=data, auth=auth) as response:
                token_data = await response.json()
                
                self.credentials.access_token = token_data['access_token']
                self.credentials.refresh_token = token_data.get('refresh_token', self.credentials.refresh_token)
                self.credentials.expires_at = datetime.now() + timedelta(seconds=token_data['expires_in'])
                
                logger.info("Fitbit token refreshed successfully")
                
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
    
    async def get_sleep_data(self, target_date: date) -> Optional[Dict]:
        """Get sleep data for a specific date"""
        date_str = target_date.strftime('%Y-%m-%d')
        endpoint = f"/user/-/sleep/date/{date_str}.json"
        
        data = await self._make_request(endpoint)
        
        if data and 'sleep' in data and data['sleep']:
            sleep_record = data['sleep'][0]  # Get main sleep record
            
            return {
                'duration_hours': sleep_record.get('duration', 0) / (1000 * 60 * 60),  # Convert ms to hours
                'efficiency': sleep_record.get('efficiency', 0),
                'quality_score': min(sleep_record.get('efficiency', 0) / 10, 10),  # Normalize to 0-10
                'start_time': sleep_record.get('startTime'),
                'end_time': sleep_record.get('endTime'),
                'raw_data': sleep_record
            }
        
        return None
    
    async def get_heart_rate_data(self, target_date: date) -> Optional[Dict]:
        """Get heart rate and HRV data for a specific date"""
        date_str = target_date.strftime('%Y-%m-%d')
        
        # Get resting heart rate
        hr_endpoint = f"/user/-/activities/heart/date/{date_str}/1d.json"
        hr_data = await self._make_request(hr_endpoint)
        
        # Get HRV data (if available)
        hrv_endpoint = f"/user/-/hrv/date/{date_str}.json"
        hrv_data = await self._make_request(hrv_endpoint)
        
        result = {}
        
        if hr_data and 'activities-heart' in hr_data:
            heart_data = hr_data['activities-heart'][0] if hr_data['activities-heart'] else {}
            if 'value' in heart_data:
                result['resting_heart_rate'] = heart_data['value'].get('restingHeartRate')
        
        if hrv_data and 'hrv' in hrv_data:
            hrv_records = hrv_data['hrv']
            if hrv_records:
                # Get the most recent HRV value
                latest_hrv = hrv_records[0]
                result['hrv'] = latest_hrv['value'].get('dailyRmssd')
        
        return result if result else None
    
    async def get_activity_data(self, target_date: date) -> Optional[Dict]:
        """Get activity data for a specific date"""
        date_str = target_date.strftime('%Y-%m-%d')
        endpoint = f"/user/-/activities/date/{date_str}.json"
        
        data = await self._make_request(endpoint)
        
        if data and 'summary' in data:
            summary = data['summary']
            
            # Calculate activity score based on steps, active minutes, and calories
            steps = summary.get('steps', 0)
            active_minutes = summary.get('veryActiveMinutes', 0) + summary.get('fairlyActiveMinutes', 0)
            calories = summary.get('caloriesOut', 0)
            
            # Normalize to 0-10 scale
            activity_score = min(
                (steps / 10000) * 3 +  # Steps component (0-3)
                (active_minutes / 60) * 4 +  # Active minutes component (0-4)
                (calories / 2500) * 3,  # Calories component (0-3)
                10
            )
            
            return {
                'steps': steps,
                'active_minutes': active_minutes,
                'calories_burned': calories,
                'activity_score': activity_score,
                'distance': summary.get('distances', [{}])[0].get('distance', 0),
                'raw_data': summary
            }
        
        return None
    
    async def get_comprehensive_data(self, target_date: date) -> HealthDataPoint:
        """Get comprehensive health data for a date"""
        sleep_data = await self.get_sleep_data(target_date)
        heart_data = await self.get_heart_rate_data(target_date)
        activity_data = await self.get_activity_data(target_date)
        
        return HealthDataPoint(
            date=target_date,
            source='fitbit',
            sleep_duration=sleep_data.get('duration_hours') if sleep_data else None,
            sleep_quality=sleep_data.get('quality_score') if sleep_data else None,
            hrv=heart_data.get('hrv') if heart_data else None,
            resting_heart_rate=heart_data.get('resting_heart_rate') if heart_data else None,
            activity_score=activity_data.get('activity_score') if activity_data else None,
            steps=activity_data.get('steps') if activity_data else None,
            calories_burned=activity_data.get('calories_burned') if activity_data else None,
            raw_data={
                'sleep': sleep_data,
                'heart': heart_data,
                'activity': activity_data
            }
        )

class OuraAPI:
    """Oura Ring API integration for health metrics"""
    
    def __init__(self, credentials: OuraCredentials = None):
        self.credentials = credentials
        self.base_url = "https://api.ouraring.com/v2"
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated request to Oura API"""
        if not self.credentials:
            raise ValueError("Oura credentials not provided")
        
        session = await self._get_session()
        headers = {
            'Authorization': f'Bearer {self.credentials.access_token}',
            'Accept': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                return await response.json()
                
        except Exception as e:
            logger.error(f"Oura API request failed: {e}")
            return {}
    
    async def get_sleep_data(self, target_date: date) -> Optional[Dict]:
        """Get sleep data from Oura"""
        start_date = target_date.strftime('%Y-%m-%d')
        end_date = target_date.strftime('%Y-%m-%d')
        
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        data = await self._make_request('/usercollection/sleep', params)
        
        if data and 'data' in data and data['data']:
            sleep_record = data['data'][0]
            
            return {
                'duration_hours': sleep_record.get('total_sleep_duration', 0) / 3600,  # Convert seconds to hours
                'efficiency': sleep_record.get('efficiency', 0),
                'quality_score': min(sleep_record.get('score', 0) / 10, 10),  # Normalize Oura score
                'rem_duration': sleep_record.get('rem_sleep_duration', 0) / 3600,
                'deep_duration': sleep_record.get('deep_sleep_duration', 0) / 3600,
                'light_duration': sleep_record.get('light_sleep_duration', 0) / 3600,
                'raw_data': sleep_record
            }
        
        return None
    
    async def get_hrv_data(self, target_date: date) -> Optional[Dict]:
        """Get HRV data from Oura"""
        start_date = target_date.strftime('%Y-%m-%d')
        end_date = target_date.strftime('%Y-%m-%d')
        
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        data = await self._make_request('/usercollection/heartrate', params)
        
        if data and 'data' in data and data['data']:
            # Oura provides detailed HRV data
            hrv_records = data['data']
            if hrv_records:
                # Calculate average HRV for the day
                hrv_values = [record.get('bpm', 0) for record in hrv_records]
                avg_hrv = sum(hrv_values) / len(hrv_values) if hrv_values else 0
                
                return {
                    'hrv': avg_hrv,
                    'resting_heart_rate': min(hrv_values) if hrv_values else None,
                    'raw_data': hrv_records
                }
        
        return None
    
    async def get_activity_data(self, target_date: date) -> Optional[Dict]:
        """Get activity data from Oura"""
        start_date = target_date.strftime('%Y-%m-%d')
        end_date = target_date.strftime('%Y-%m-%d')
        
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        data = await self._make_request('/usercollection/daily_activity', params)
        
        if data and 'data' in data and data['data']:
            activity_record = data['data'][0]
            
            # Oura provides activity scores directly
            activity_score = min(activity_record.get('score', 0) / 10, 10)  # Normalize to 0-10
            
            return {
                'steps': activity_record.get('steps', 0),
                'calories_burned': activity_record.get('active_calories', 0),
                'activity_score': activity_score,
                'active_minutes': activity_record.get('high_activity_time', 0) / 60,  # Convert to minutes
                'raw_data': activity_record
            }
        
        return None
    
    async def get_readiness_data(self, target_date: date) -> Optional[Dict]:
        """Get readiness/recovery data from Oura"""
        start_date = target_date.strftime('%Y-%m-%d')
        end_date = target_date.strftime('%Y-%m-%d')
        
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        data = await self._make_request('/usercollection/daily_readiness', params)
        
        if data and 'data' in data and data['data']:
            readiness_record = data['data'][0]
            
            return {
                'readiness_score': min(readiness_record.get('score', 0) / 10, 10),  # Normalize to 0-10
                'recovery_score': min(readiness_record.get('score', 0) / 10, 10),
                'raw_data': readiness_record
            }
        
        return None
    
    async def get_comprehensive_data(self, target_date: date) -> HealthDataPoint:
        """Get comprehensive health data for a date"""
        sleep_data = await self.get_sleep_data(target_date)
        hrv_data = await self.get_hrv_data(target_date)
        activity_data = await self.get_activity_data(target_date)
        readiness_data = await self.get_readiness_data(target_date)
        
        return HealthDataPoint(
            date=target_date,
            source='oura',
            sleep_duration=sleep_data.get('duration_hours') if sleep_data else None,
            sleep_quality=sleep_data.get('quality_score') if sleep_data else None,
            hrv=hrv_data.get('hrv') if hrv_data else None,
            resting_heart_rate=hrv_data.get('resting_heart_rate') if hrv_data else None,
            activity_score=activity_data.get('activity_score') if activity_data else None,
            steps=activity_data.get('steps') if activity_data else None,
            calories_burned=activity_data.get('calories_burned') if activity_data else None,
            recovery_score=readiness_data.get('recovery_score') if readiness_data else None,
            raw_data={
                'sleep': sleep_data,
                'hrv': hrv_data,
                'activity': activity_data,
                'readiness': readiness_data
            }
        )

class HealthDataAggregator:
    """Aggregate health data from multiple sources"""
    
    def __init__(self):
        self.fitbit_api = None
        self.oura_api = None
        self.manual_data = {}  # Store manually entered data
    
    def configure_fitbit(self, credentials: FitbitCredentials):
        """Configure Fitbit API"""
        self.fitbit_api = FitbitAPI(credentials)
    
    def configure_oura(self, credentials: OuraCredentials):
        """Configure Oura API"""
        self.oura_api = OuraAPI(credentials)
    
    def add_manual_data(self, user_id: str, data_point: HealthDataPoint):
        """Add manually entered health data"""
        if user_id not in self.manual_data:
            self.manual_data[user_id] = []
        self.manual_data[user_id].append(data_point)
    
    async def get_health_data(self, user_id: str, target_date: date, 
                            preferred_source: str = 'auto') -> Optional[HealthDataPoint]:
        """Get health data for a user and date, preferring specified source"""
        
        results = {}
        
        # Try Fitbit if available
        if self.fitbit_api and (preferred_source == 'fitbit' or preferred_source == 'auto'):
            try:
                results['fitbit'] = await self.fitbit_api.get_comprehensive_data(target_date)
            except Exception as e:
                logger.error(f"Fitbit data retrieval failed: {e}")
        
        # Try Oura if available
        if self.oura_api and (preferred_source == 'oura' or preferred_source == 'auto'):
            try:
                results['oura'] = await self.oura_api.get_comprehensive_data(target_date)
            except Exception as e:
                logger.error(f"Oura data retrieval failed: {e}")
        
        # Check manual data
        manual_data = self._get_manual_data(user_id, target_date)
        if manual_data:
            results['manual'] = manual_data
        
        # Combine data sources intelligently
        return self._combine_data_sources(results, preferred_source)
    
    def _get_manual_data(self, user_id: str, target_date: date) -> Optional[HealthDataPoint]:
        """Get manual data for user and date"""
        if user_id in self.manual_data:
            for data_point in self.manual_data[user_id]:
                if data_point.date == target_date:
                    return data_point
        return None
    
    def _combine_data_sources(self, results: Dict[str, HealthDataPoint], 
                            preferred_source: str) -> Optional[HealthDataPoint]:
        """Intelligently combine data from multiple sources"""
        if not results:
            return None
        
        # If preferred source is available and has good data, use it
        if preferred_source in results:
            primary_data = results[preferred_source]
            if self._is_complete_data(primary_data):
                return primary_data
        
        # Otherwise, combine the best available data
        combined_data = HealthDataPoint(
            date=list(results.values())[0].date,
            source='combined'
        )
        
        # Priority order for data sources
        source_priority = ['oura', 'fitbit', 'manual']
        
        for field in ['sleep_duration', 'sleep_quality', 'hrv', 'activity_score', 
                     'steps', 'calories_burned', 'recovery_score']:
            for source in source_priority:
                if source in results:
                    value = getattr(results[source], field)
                    if value is not None:
                        setattr(combined_data, field, value)
                        break
        
        # Calculate missing values if possible
        combined_data = self._calculate_missing_values(combined_data)
        
        return combined_data
    
    def _is_complete_data(self, data_point: HealthDataPoint) -> bool:
        """Check if data point has most essential metrics"""
        essential_fields = ['sleep_quality', 'activity_score']
        return all(getattr(data_point, field) is not None for field in essential_fields)
    
    def _calculate_missing_values(self, data_point: HealthDataPoint) -> HealthDataPoint:
        """Calculate missing values based on available data"""
        
        # Calculate recovery score if missing
        if data_point.recovery_score is None:
            recovery_components = []
            if data_point.sleep_quality is not None:
                recovery_components.append(data_point.sleep_quality * 0.4)
            if data_point.hrv is not None:
                # Normalize HRV to 0-10 scale
                normalized_hrv = min(data_point.hrv / 10, 10)
                recovery_components.append(normalized_hrv * 0.3)
            if data_point.activity_score is not None:
                # Lower activity can indicate better recovery
                recovery_components.append((10 - data_point.activity_score) * 0.3)
            
            if recovery_components:
                data_point.recovery_score = sum(recovery_components)
        
        # Calculate stress level if missing
        if data_point.stress_level is None and data_point.hrv is not None:
            # Lower HRV typically indicates higher stress
            if data_point.hrv < 30:
                data_point.stress_level = 8.0
            elif data_point.hrv < 50:
                data_point.stress_level = 5.0
            else:
                data_point.stress_level = 3.0
        
        return data_point
    
    async def get_historical_data(self, user_id: str, days: int = 7) -> List[HealthDataPoint]:
        """Get historical health data for multiple days"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days-1)
        
        historical_data = []
        current_date = start_date
        
        while current_date <= end_date:
            data_point = await self.get_health_data(user_id, current_date)
            if data_point:
                historical_data.append(data_point)
            current_date += timedelta(days=1)
        
        return historical_data

# Global health data aggregator
health_aggregator = HealthDataAggregator()

# Demo functions for testing without real API credentials
def create_demo_health_data(target_date: date = None) -> HealthDataPoint:
    """Create demo health data for testing"""
    if target_date is None:
        target_date = date.today()
    
    # Generate realistic demo data
    import random
    
    return HealthDataPoint(
        date=target_date,
        source='demo',
        sleep_duration=random.uniform(6.5, 9.0),
        sleep_quality=random.uniform(6.0, 9.5),
        hrv=random.uniform(25, 75),
        resting_heart_rate=random.uniform(50, 80),
        activity_score=random.uniform(3.0, 9.0),
        steps=random.randint(3000, 15000),
        calories_burned=random.uniform(1800, 3200),
        stress_level=random.uniform(2.0, 8.0),
        recovery_score=random.uniform(5.0, 9.5)
    )

def get_demo_historical_data(days: int = 7) -> List[HealthDataPoint]:
    """Get demo historical data for testing"""
    historical_data = []
    for i in range(days):
        target_date = date.today() - timedelta(days=i)
        historical_data.append(create_demo_health_data(target_date))
    
    return historical_data[::-1]  # Return in chronological order