#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track loader for loading custom tracks from JSON files
"""
import json
import numpy as np
import os

def load_track_from_json(json_path):
    """
    Load track waypoints and parameters from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file
        
    Returns:
        tuple: (waypoints, scale, track_width)
            - waypoints: numpy array of shape (N, 2) containing waypoint coordinates
            - scale: float, scaling factor
            - track_width: float, width of the track in meters
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Track file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    waypoints = np.array(data.get('waypoints', []))
    scale = data.get('scale', 2.0)
    track_width = data.get('trackWidth', 7.5)
    
    if len(waypoints) < 3:
        raise ValueError("Track must have at least 3 waypoints")
    
    return waypoints, scale, track_width

def get_track_from_json_or_default(json_path=None):
    """
    Load track from JSON file if provided, otherwise use default waypoints.
    
    Args:
        json_path (str, optional): Path to JSON file. If None, uses default track.
        
    Returns:
        tuple: (waypoints, scale, track_width)
    """
    if json_path and os.path.exists(json_path):
        print(f"📂 Loading custom track from: {json_path}")
        try:
            waypoints, scale, track_width = load_track_from_json(json_path)
            print(f"✅ Loaded {len(waypoints)} waypoints (scale={scale}, width={track_width}m)")
            return waypoints, scale, track_width
        except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"⚠️ Failed to load track: {e}")
            print("   Using default track instead.")
    
    # Default track (from original g_save.py)
    print("📍 Using default track")
    waypoints = np.array([
        [0, -20], [-30, -20], [-20, 20], [-10, 30], [10, 40],
        [80, 80], [100, 70], [90, 50], [110, 30], [90, -10],
        [100, -50], [60, -70], [20, -60], [0, -40],
    ])
    scale = 2.0
    track_width = 7.5
    
    return waypoints, scale, track_width

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python track_loader.py <track.json>")
        print("\nThis script loads and validates a track JSON file.")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    try:
        waypoints, scale, track_width = load_track_from_json(json_path)
        print(f"\n✅ Successfully loaded track!")
        print(f"   Waypoints: {len(waypoints)}")
        print(f"   Scale: {scale}")
        print(f"   Track width: {track_width}m")
        print(f"\nFirst 5 waypoints:")
        for i, wp in enumerate(waypoints[:5]):
            print(f"   {i}: ({wp[0]:.1f}, {wp[1]:.1f})")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
