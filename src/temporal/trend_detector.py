# ============================================================================
# FILE 1: src/medical_ingestion/temporal/trend_detector.py
# ============================================================================
"""
Trend Detector - Identifies patterns in lab values over time

Detects:
- Linear trends (rising, falling, stable)
- Rate of change (acute vs chronic)
- Trend reversals
- Cyclical patterns

This is the foundation for temporal analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass


@dataclass
class TrendPoint:
    """Single data point in a trend"""
    date: datetime
    value: float
    unit: str


@dataclass
class DetectedTrend:
    """A detected trend in lab values"""
    test_name: str
    direction: str  # "rising", "falling", "stable"
    rate: float  # Change per day
    start_date: datetime
    end_date: datetime
    start_value: float
    end_value: float
    total_change: float
    percent_change: float
    confidence: float
    data_points: int


class TrendDetector:
    """
    Detects trends in time-series lab data.
    
    Uses simple linear regression and statistical tests
    to identify significant trends.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_trend(
        self,
        test_name: str,
        historical_values: List[Dict[str, Any]],
        current_value: float,
        current_date: datetime
    ) -> Optional[DetectedTrend]:
        """
        Detect trend in lab values.
        
        Args:
            test_name: Name of test
            historical_values: List of {date, value} dicts
            current_value: Current value
            current_date: Current date
            
        Returns:
            DetectedTrend if significant trend found, None otherwise
        """
        # Need at least 2 points to detect trend
        if len(historical_values) < 2:
            return None
        
        # Convert to TrendPoints
        points = self._prepare_data_points(
            historical_values,
            current_value,
            current_date
        )
        
        if len(points) < 3:
            return None
        
        # Calculate linear trend
        slope, intercept = self._calculate_linear_fit(points)
        
        # Determine direction
        direction = self._determine_direction(slope)
        
        # Calculate statistics
        start_point = points[0]
        end_point = points[-1]
        
        total_change = end_point.value - start_point.value
        percent_change = (total_change / start_point.value * 100) if start_point.value != 0 else 0
        
        days_elapsed = (end_point.date - start_point.date).days
        rate = total_change / days_elapsed if days_elapsed > 0 else 0
        
        # Calculate confidence (R-squared)
        confidence = self._calculate_r_squared(points, slope, intercept)
        
        # Only return if trend is significant
        if confidence < 0.5:  # Low correlation
            return None
        
        if abs(percent_change) < 5:  # Less than 5% change
            return None
        
        return DetectedTrend(
            test_name=test_name,
            direction=direction,
            rate=rate,
            start_date=start_point.date,
            end_date=end_point.date,
            start_value=start_point.value,
            end_value=end_point.value,
            total_change=total_change,
            percent_change=percent_change,
            confidence=confidence,
            data_points=len(points)
        )
    
    def _prepare_data_points(
        self,
        historical_values: List[Dict],
        current_value: float,
        current_date: datetime
    ) -> List[TrendPoint]:
        """Convert raw data to TrendPoints, sorted by date"""
        points = []
        
        for item in historical_values:
            points.append(TrendPoint(
                date=item['date'],
                value=item['value'],
                unit=item.get('unit', '')
            ))
        
        # Add current value
        points.append(TrendPoint(
            date=current_date,
            value=current_value,
            unit=points[0].unit if points else ''
        ))
        
        # Sort by date
        points.sort(key=lambda p: p.date)
        
        return points
    
    def _calculate_linear_fit(
        self,
        points: List[TrendPoint]
    ) -> Tuple[float, float]:
        """
        Calculate linear regression (y = mx + b).
        
        Returns (slope, intercept)
        """
        n = len(points)
        
        # Convert dates to days since first point
        base_date = points[0].date
        x_values = [(p.date - base_date).days for p in points]
        y_values = [p.value for p in points]
        
        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        # Calculate slope
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Calculate intercept
        intercept = y_mean - slope * x_mean
        
        return slope, intercept
    
    def _determine_direction(self, slope: float) -> str:
        """Determine trend direction from slope"""
        if slope > 0.01:
            return "rising"
        elif slope < -0.01:
            return "falling"
        else:
            return "stable"
    
    def _calculate_r_squared(
        self,
        points: List[TrendPoint],
        slope: float,
        intercept: float
    ) -> float:
        """
        Calculate R-squared (coefficient of determination).
        
        Measures how well the linear fit explains the data.
        0 = no correlation, 1 = perfect correlation
        """
        base_date = points[0].date
        x_values = [(p.date - base_date).days for p in points]
        y_values = [p.value for p in points]
        
        # Predicted values
        y_pred = [slope * x + intercept for x in x_values]
        
        # Mean of actual values
        y_mean = sum(y_values) / len(y_values)
        
        # Total sum of squares
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        
        # Residual sum of squares
        ss_res = sum((y - y_p) ** 2 for y, y_p in zip(y_values, y_pred))
        
        # R-squared
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return max(0, min(1, r_squared))  # Clamp to [0, 1]