import os
import json
import numpy as np

from PIL import Image, ImageDraw

from segmodels.utils import LBL_RAIL_RAISED

class osdar23gt:
    def __init__(self, dataset_root, scene, cls_value=LBL_RAIL_RAISED, img_size=(2504, 4112)):
        self.img_size = img_size
        self.cls_value = cls_value
        self.lbl_file = os.path.join(dataset_root, scene, scene + "_labels.json")

    def predict(self, img_filename, rasterWidth=10):
        frame_name = str(int(img_filename.split('_')[0]))

        lines = []

        with open(self.lbl_file, 'r') as data_file:
            scene = json.load(data_file)

            objs = scene['openlabel']['objects']
            track_ids = []

            for obj in objs:
                if objs[obj]["type"] == "track":
                    track_ids.append(obj)

            tracks = scene['openlabel']['frames'][frame_name]['objects']
            for track_id in track_ids:
                if track_id in tracks.keys():
                    sensors = tracks[track_id]['object_data']['poly2d']
                    
                    for sensor in sensors:
                        if sensor['name'] == 'rgb_highres_center__poly2d__track':
                            pixels = []
                            pnts = sensor['val']
                            for i in range(0, len(pnts), 2):
                                pixels.append([pnts[i], pnts[i+1]])

                            lines.append(pixels)

        gt_msk = np.zeros(self.img_size, dtype=np.uint8)

        gt_msk_pil = Image.fromarray(gt_msk)
        draw = ImageDraw.Draw(gt_msk_pil)

        for line in lines:
            for i in range(len(line)-1):
                draw.line([(int(line[i][0]), int(line[i][1]) ), ( int(line[i+1][0]), int(line[i+1][1]) ) ], fill=self.cls_value, width=rasterWidth)

        return np.array(gt_msk_pil)
