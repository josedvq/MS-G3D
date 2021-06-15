import os
import glob
from pathlib import Path
import json
import pickle
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset


num_joint = 18
max_frame = 300
num_person_out = 2
num_person_in = 5


class Feeder_conflab(Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    # Joint index:
    # {0,  "Nose"}
    # {1,  "Neck"},
    # {2,  "RShoulder"},
    # {3,  "RElbow"},
    # {4,  "RWrist"},
    # {5,  "LShoulder"},
    # {6,  "LElbow"},
    # {7,  "LWrist"},
    # {8,  "RHip"},
    # {9,  "RKnee"},
    # {10, "RAnkle"},
    # {11, "LHip"},
    # {12, "LKnee"},
    # {13, "LAnkle"},
    # {14, "REye"},
    # {15, "LEye"},
    # {16, "REar"},
    # {17, "LEar"},
    Arguments:
        data_path: the path to folder with skeletons in COCO JSON format
        label_path: the path to label
        window_size: The length of the output sequence
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 window_size=-1):
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size

        self.load_data()
        self.load_labels()

    def load_data(self):
        # load file list
        self.segments = [] # [{vid: int, seg: int, tracks: {[pid]: [{kp: X, occl: Y}]]}}]

        for seg_file in tqdm(Path(self.data_path).glob('*.json')):
            parts = os.path.basename(seg_file).split('_')
            vid = parts[1][-1:]
            seg = parts[2][-1:]

            seg_dict = {'vid':  vid, 'seg': seg, 'tracks': {}}
            with open(seg_file) as f:
                coco_json = json.load(f)
                for frame_skeletons in coco_json['annotations']['skeletons']:
                    for fs in frame_skeletons.values():
                        pid = fs['id']

                        if pid not in seg_dict['tracks']:
                            seg_dict['tracks'][pid] = []
                        
                        seg_dict['tracks'][pid].append({
                            'frame': fs['image_id'],
                            'kp': fs['keypoints'],
                            'occl': fs['occluded']
                        })
            for pid, track in seg_dict['tracks'].items():
                new_track = [[e['frame'], *e['kp'], *e['occl']] for e in track]
                seg_dict['tracks'][pid] = np.array(new_track)
            self.segments.append(seg_dict)

        # # output data shape (N, C, T, V, M)
        # self.N = len(self.sample_name)  # number of examples
        # self.C = 3  # x, y, conf
        # self.T = max_frame  # frame / time
        # self.V = num_joint  # joint

    def parse_fname(self, fname):
        if '-' in fname:
            person_part = fname.split('-')[0]
            conf_part = fname.split('-')[1]
            pid = int(person_part[6:])
            subm = int(conf_part.split('_')[1])
            return pid, 'conf', subm
        else:
            person_part = fname.split('_')[0]
            subm = int(fname.split('_')[1])
            pid = int(person_part[6:])
            return pid, 'ss', subm

    def load_labels(self):
        self.labels = {} # {[segment]: {[pid]: {ss: dataframe, conf: dataframe}}}
        for seg_folder in tqdm(Path(self.label_path).glob('*')):
            seg_name = os.path.basename(seg_folder)
            self.labels[seg_name] = {}
            for fpath in seg_folder.glob('*.csv'):
                fname = os.path.basename(fpath).split('.')[0]
                if '_av_' in fname or 'Sample' in fname:
                    continue
                pid, signal, subm= self.parse_fname(fname)
                if pid not in self.labels[seg_name]:
                    self.labels[seg_name][pid] = {'ss': None, 'conf': None}

                d = pd.read_csv(fpath)
                self.labels[seg_name][pid][signal] = d

    def split_data(self, window_len=180):
        '''
        Splits data into examples.
        '''
        for segment in self.segments:
            ss = self.labels[f'vid{segment["vid"]}_seg{segment["seg"]}']
            for pid, track in segment['tracks'].items():
                assert track[-1,0] == len(track)-1

                split_indices = np.arange(window_len, len(track), window_len)
                track_examples = np.split(track, split_indices)
                if len(track_examples[-1]) != window_len:
                    track_examples.pop()
                
                # get the label
                pid_ss = ss[pid]['ss']
                for example in track_examples:
                    ini_time = 59.94 * example[0, 0]
                    end_time = 59.94 * example[-1, 0]
                    example_ss = pid_ss[pid_ss['media_time'] > ini_time & pid_ss['media_time'] < end_time]
                    print(len(example_ss))

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[1:2] = -data_numpy[1:2]
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        return data_numpy, label


def gendata(data_path, label_path,
            out_folder,
            max_frame=max_frame):
    feeder = Feeder_conflab(
        data_path=data_path,
        label_path=label_path,
        window_size=max_frame)

    # sample_name = feeder.sample_name
    # sample_label = []

    # fp = np.zeros((len(sample_name), 3, max_frame, num_joint, num_person_out), dtype=np.float32)

    # for i, s in enumerate(tqdm(sample_name)):
    #     data, label = feeder[i]
    #     fp[i, :, 0:data.shape[1], :, :] = data
    #     sample_label.append(label)

    # with open(label_out_path, 'wb') as f:
    #     pickle.dump((sample_name, list(sample_label)), f)

    # np.save(data_out_path, fp)

    # separate in train and test sets.

    data_out_path = '{}/{}_data_joint.npy'.format(out_folder, 'train')
    label_out_path = '{}/{}_label.pkl'.format(out_folder, 'train')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Conflab dataset skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='../data/conflab_raw/coco')
    parser.add_argument(
        '--label_path', default='../data/conflab_raw/speaking_status')
    parser.add_argument(
        '--out_folder', default='../data/conflab')
    arg = parser.parse_args()

    if not os.path.exists(arg.out_folder):
        os.makedirs(arg.out_folder)
    

    gendata(arg.data_path, arg.label_path, arg.out_folder)
