import os
import glob
from pathlib import Path
import json
import pickle
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from torch.utils import data
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


num_joint = 18
max_frame = 180
num_person = 2

bt2 = datetime(2019, 10, 24, 16, 49, 36, int(1000000 * 58/59.94)) # start timecode of vid2 : 14:49:36:58
bt3 = datetime(2019, 10, 24, 17, 7, 13, int(1000000 * 58/59.94)) # start timecode of vid3 : 15:07:13:58
timecodes = {
    'vid2_seg8': bt2 + timedelta(minutes=14),
    'vid2_seg9': bt2 + timedelta(minutes=16),
    'vid3_seg1': bt3 + timedelta(minutes=0),
    'vid3_seg2': bt3 + timedelta(minutes=2),
    'vid3_seg3': bt3 + timedelta(minutes=4),
    'vid3_seg4': bt3 + timedelta(minutes=6),
    'vid3_seg5': bt3 + timedelta(minutes=8),
    'vid3_seg6': bt3 + timedelta(minutes=10),
}


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
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 window_size=-1):
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size
        self.examples = []
        self.accel_ds = None

        self.load_data()
        self.load_labels()

    def load_data(self):
        # load file list
        self.segments = [] # [{vid: int, seg: int, tracks: {[pid]: [{kp: X, occl: Y}]]}}]

        for seg_file in tqdm(Path(self.data_path).glob('*.json')):
            parts = os.path.basename(seg_file).split('_')
            cam = parts[0][-1:]
            vid = parts[1][-1:]
            seg = parts[2][-1:]

            seg_dict = {'cam': cam, 'vid':  vid, 'seg': seg, 'tracks': {}}
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
                seg_dict['tracks'][pid] = np.array(new_track, dtype=np.float64)
            self.segments.append(seg_dict)

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

    def make_examples(self, window_len=180):
        '''
        Splits data into examples.
        '''
        self.examples = []
        for segment in tqdm(self.segments):
            ss_name = f'vid{segment["vid"]}_seg{segment["seg"]}'
            ss = self.labels[ss_name]
            for pid, track in segment['tracks'].items():
                assert track[-1,0] == len(track)-1

                split_indices = np.arange(window_len, len(track), window_len)
                track_examples = np.split(track, split_indices)
                if len(track_examples[-1]) != window_len:
                    track_examples.pop()
                
                # get the label
                if pid not in ss:
                    print(f'PID {pid} not found in ss {ss_name}')
                    continue
                pid_ss = ss[pid]['ss']
                for example in track_examples:
                    assert len(example) == window_len
                    ini_time = example[0, 0] / 59.94
                    end_time = example[-1, 0] / 59.94
                    example_ss = pid_ss.loc[(pid_ss['media_time'] >= ini_time) & (pid_ss['media_time'] < end_time)]
                    example_ss = example_ss[example_ss['index'] != 0] # remove skipped frames
                    label = (example_ss['data0'].mean(skipna=True) > 0.5)
                    self.examples.append({
                        'cam': segment['cam'],
                        'vid': segment['vid'],
                        'seg': segment['seg'],
                        'pid': pid,
                        'ini': ini_time,
                        'end': end_time,
                        'ss': example_ss,
                        'ss_mean': example_ss['data0'].mean(skipna=True),
                        'label': label, 
                        'data': example})

        # output data shape (N, C, T, V, M)
        self.N = len(self.examples)  # number of examples
        self.C = 3  # x, y, conf
        self.T = window_len  # frame / time
        self.V = num_joint  # joint
        self.num_person = num_person

    def load_accel(self, accel_path):
        self.accel = {}
        for accel_path in Path(accel_path).glob('*.csv'):
            pid = int(os.path.basename(accel_path).split('_')[0])
            self.accel[pid] = pd.read_csv(accel_path, index_col=0)

    def make_accel_dataset(self):
        self.accel_ds = np.zeros((len(self.examples), 3, 165))
        for i, ex in enumerate(tqdm(self.examples)):
            if ex['pid'] not in self.accel:
                print(f'No accel for pid {ex["pid"]}')
                continue
            df = self.accel[ex['pid']]
            df['time'] = pd.to_datetime(df['time'])

            seg_dt = timecodes[f'vid{ex["vid"]}_seg{ex["seg"]}']
            ini_dt = seg_dt + timedelta(seconds=ex['ini'])
            end_dt = seg_dt + timedelta(seconds=ex['end'])

            ex_accel = df.loc[(df['time'] >= ini_dt) & (df['time'] < end_dt)]
            ex_accel = ex_accel.loc[:, ['X', 'Y', 'Z']].to_numpy()
            ex_accel = ex_accel[0:165, :].transpose()
            self.accel_ds[i, :, 0:ex_accel.shape[1]] = ex_accel

        return self.accel_ds

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        example = self.examples[index]

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person))
        data_numpy[0, :, 0:17, 0] = example['data'][:,1:1+34:2]
        data_numpy[1, :, 0:17, 0] = example['data'][:,2:2+34:2]
        # data_numpy[2, :, :, 0] = example['data'][:,35:] # oclussion
        data_numpy[2, :, :, 0] = 0.5

        # map from conflab to kinetics joints
        # Kinetics (target) joint index:
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
        data_numpy[:,:,:,0] = data_numpy[:,:,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0, 0, 0],0]
        data_numpy[2, :, 14:18, 0] = 0 # 14-18 are not in conflab

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[1:2] = -data_numpy[1:2]

        data_numpy = np.nan_to_num(data_numpy) # new: convert NaN to zero
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        return data_numpy, example['label']

def gendata_from_feeder(feeder, out_folder):
    val_size = 0.2

    subjects = list(set([ex['pid'] for ex in feeder.examples]))
    train_subjects, val_subjects = train_test_split(subjects, test_size=val_size, random_state=22, shuffle=True)

    idxs = {}
    idxs['train'] = [i for i, ex in enumerate(feeder.examples) if ex['pid'] in train_subjects]
    idxs['val'] = [i for i, ex in enumerate(feeder.examples) if ex['pid'] in val_subjects]
    print(f'train sz: {len(idxs["train"])}, val sz: {len(idxs["val"])}')
    # idxs['train'], idxs['val'] = train_test_split(range(0, len(feeder)), test_size=val_size, random_state=22, shuffle=True)

    for p in ['val', 'train']:
        sample_label = []
        fp = np.zeros((len(idxs[p]), 3, max_frame, num_joint, num_person), dtype=np.float32)
        for j, i in enumerate(idxs[p]):
            data, label = feeder[i]
            fp[j, :, 0:data.shape[1], :, :] = data
            sample_label.append(label)

        data_out_path = '{}/{}_data_joint.npy'.format(out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(out_folder, p)
        sample_name = [str(i) for i in range(0, len(idxs[p]))]
        with open(label_out_path, 'wb') as f:
            pickle.dump((sample_name, list(sample_label)), f)

        np.save(data_out_path, fp)

def gen_accel_data_from_feeder(feeder, out_folder):
    val_size = 0.2

    subjects = list(set([ex['pid'] for ex in feeder.examples]))
    train_subjects, val_subjects = train_test_split(subjects, test_size=val_size, random_state=22, shuffle=True)

    idxs = {}
    idxs['train'] = [i for i, ex in enumerate(feeder.examples) if ex['pid'] in train_subjects]
    idxs['val'] = [i for i, ex in enumerate(feeder.examples) if ex['pid'] in val_subjects]
    print(f'train sz: {len(idxs["train"])}, val sz: {len(idxs["val"])}')
    accel = feeder.make_accel_dataset()

    for p in ['val', 'train']:
        sample_label = []
        fp = np.zeros((len(idxs[p]), 3, 165), dtype=np.float32)
        for j, i in enumerate(idxs[p]):
            label = feeder.examples[i]['label']
            fp[j, :, :] = accel[i, :, :]
            sample_label.append(label)

        data_out_path = '{}/accel3s_{}.npy'.format(out_folder, p)
        label_out_path = '{}/accel3s_{}_label.pkl'.format(out_folder, p)
        sample_name = [str(i) for i in range(0, len(idxs[p]))]
        with open(label_out_path, 'wb') as f:
            pickle.dump((sample_name, list(sample_label)), f)

        np.save(data_out_path, fp)

def gendata_random(data_path, label_path,
            out_folder):
    feeder = Feeder_conflab(
        data_path=data_path,
        label_path=label_path,
        window_size=180)
    feeder.make_examples()
    return gendata_from_feeder(feeder, out_folder)
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Conflab dataset skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='../data/conflab_raw/coco')
    parser.add_argument(
        '--label_path', default='../data/conflab_raw/speaking_status')
    parser.add_argument(
        '--out_folder', default='../data/conflab_random')
    arg = parser.parse_args()

    if not os.path.exists(arg.out_folder):
        os.makedirs(arg.out_folder)
    
    np.random.seed(22)
    gendata_random(arg.data_path, arg.label_path, arg.out_folder)
