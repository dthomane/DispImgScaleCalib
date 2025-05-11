from dataclasses import dataclass, asdict
import json

@dataclass
class CalibParams:
    R: list
    T: list
    s: float
    f: float
    cx: float
    cy: float
    acc: float
    min_disp: float
    max_dist: float

    def __init__(self):
        self.R = []
        self.T = []
        self.s = 1
        self.f = 1
        self.cx = 0
        self.cy = 0
        self.acc = 0
        self.min_disp = 20
        self.max_dist = 50

    def save(self, file):
        with open(file, 'w') as f:
            json.dump(asdict(self), f)

    def load(self, file):
        f = open(file)
        data = json.load(f)

        self.R = data['R']
        self.T = data['T']
        self.s = data['s']
        self.f = data['f']
        self.cx = data['cx']
        self.cy = data['cy']


@dataclass
class CalibConstants:

    min_disp: float
        
    accuracy_thresh: float
    lower_acc_if_no_res: float
    min_acc_thresh: float
    
    f: float
    cx: float
    cy: float

    def __init__(self):
        self.min_disp = 20
        self.max_dist = 50

        self.lower_dist_if_no_res = 1
        self.min_dist_if_no_res = 5

        self.higher_disp_if_no_res = 1
        self.max_disp_if_no_res = 25
        
        self.accuracy_thresh = 0.9
        self.lower_acc_if_no_res = 0.01
        self.min_acc_thresh = 0.5

        self.f = 7267.95450880415
        self.cx = 2056.049238502414
        self.cy = 1232.862908875167