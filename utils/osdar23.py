from dataclasses import dataclass
import numpy as np

from scipy.spatial.transform import Rotation as Rot

calib_scenes = [
    '3_fire_site_3.1', # 100 forward
    '3_fire_site_3.2', # 9 forward
    '3_fire_site_3.3', # 9 backward
    '3_fire_site_3.4', # 9 backward
    '9_station_ruebenkamp_9.1', # 9 forward
    '9_station_ruebenkamp_9.2', # 9 forward
    '9_station_ruebenkamp_9.3', # 9 forward
    '9_station_ruebenkamp_9.4', # 9 forward
    '9_station_ruebenkamp_9.7', # 9 forward
    '15_construction_vehicle_15.1', # 11 forward
    #
    '6_station_klein_flottbek_6.2', ## 100 forward, ##maybe a few##
    '8_station_altona_8.1', ## 100 steady
    '8_station_altona_8.2', ## 100 steady
    '8_station_altona_8.3', ## 10 steady
    '9_station_ruebenkamp_9.5', # 10 steady
    #
    '10_station_suelldorf_10.1',#ok
    #
    '12_vegetation_steady_12.1', # 98 steady
    '14_signals_station_14.3', # 10 forward
    '21_station_wedel_21.1', # 100 steady
    '21_station_wedel_21.2', # 98 steady
    '21_station_wedel_21.3' # 39 steady
]

suboptimal_scenes = [
    '13_station_ohlsdorf_13.1',
    '14_signals_station_14.1',
    '19_vegetation_curve_19.1'
]

scenes =  [
    '1_calibration_1.1', # on switch, 9 steady
    '1_calibration_1.2', # on switch, 100 forward
    '2_station_berliner_tor_2.1', # on left curve, 10 forward
    '3_fire_site_3.1', # straight, 100 forward
    '3_fire_site_3.2', # straight, with smoke, 9 forward
    '3_fire_site_3.3', # straight, 9 backward
    '3_fire_site_3.4', # straight, 9 backward
    '4_station_pedestrian_bridge_4.1', # first straight right curve comming up, 9 steady
    '4_station_pedestrian_bridge_4.2', # first straight right curve comming up, 9 steady
    '4_station_pedestrian_bridge_4.3', # first straight right curve comming up, 98 steady
    '4_station_pedestrian_bridge_4.4', # first straight right curve comming up, 98 staedy
    '4_station_pedestrian_bridge_4.5', # first straight right curve comming up, 76 slightly forward appr 1m
    '5_station_bergedorf_5.1', # slightly straight than on left curve, 100 forward
    '5_station_bergedorf_5.2', # on left curve, straight comming up, 10 forward
    '6_station_klein_flottbek_6.1', # on switch, than straight, than right curve, 10 forward
    '6_station_klein_flottbek_6.2',  # straight, right curve comming up, 100 forward
    '7_approach_underground_station_7.1', # on right curve, switch comming up, 10 forward
    '7_approach_underground_station_7.2', # on right curve, switch comming up, 10 forward
    '7_approach_underground_station_7.3', # on switch, straight, than another switch, 10 forward
    '8_station_altona_8.1', # underground straight, switch far away comming up, 100 steady
    '8_station_altona_8.2', # underground straight, switch far away comming up, 100 steady
    '8_station_altona_8.3', # underground straight, switch far away comming up, 10 steady
    '9_station_ruebenkamp_9.1', # straight, very slightly left curve commin up, 10 forward
    '9_station_ruebenkamp_9.2', # straight, very slightly right curve coming up, 10 forward
    '9_station_ruebenkamp_9.3', # straight, switch far away comming up, 10 forward
    '9_station_ruebenkamp_9.4', # straight, switch far away comming up, 10 forward
    '9_station_ruebenkamp_9.5', # straight, left curve comming up, 10 steady
    '9_station_ruebenkamp_9.6', # left-right-S-shape, 10 backward
    '9_station_ruebenkamp_9.7', # straight, very slightly left curved, 10 forward
    '10_station_suelldorf_10.1', # on left curve, far away switch comming up, 10 backward
    '11_main_station_11.1', # on left curve, 10 forward
    '12_vegetation_steady_12.1', # straight, 98 steady
    '13_station_ohlsdorf_13.1', # on right curved switch, 10 forward
    '14_signals_station_14.1', # on right curved switch, 10 forward
    '14_signals_station_14.2', # on left switch merging track, 10 steady
    '14_signals_station_14.3', # straight, 10 forward
    '15_construction_vehicle_15.1', # straight, 11 forward
    '16_under_bridge_16.1', # on right curve, 10 steady
    '17_signal_bridge_17.1', # on switch, 10 forward
    '18_vegetation_switch_18.1', # on left curve, switch comming up, 10 forward
    '19_vegetation_curve_19.1', # on left curve, 10 forward
    '20_vegetation_squirrel_20.1', # on right curve, 10 forward
    '21_station_wedel_21.1', # short straight than end, 100 steady ###
    '21_station_wedel_21.2', # short straight than end, 98 steady ###
    '21_station_wedel_21.3'  # short straight than end, 39 steady ###
]

@dataclass
class osdar23extrinsics:
    R0: np.array
    R: np.array
    T: np.array

    def __init__(self):
        self.R0 = np.array([[0,0,1],[-1,0,0],[0,-1,0]])

        q = np.array([-0.00313306, 0.0562995, 0.00482918, 0.998397])
        self.R = Rot.from_quat(q).as_matrix()
        
        self.T = np.array([0.0801578, -0.332862, 3.50982])