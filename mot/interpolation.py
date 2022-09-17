import sys
import argparse
import numpy as np
import os
import glob

sys.path.append('.')


def make_parser():
    parser = argparse.ArgumentParser("Interpolation!")
    parser.add_argument("txt_path", default="", help="path to tracking result path in MOTChallenge format")
    parser.add_argument("save_path", default=None, help="save result path, none for override")
    parser.add_argument("--n_min", type=int, default=5, help="minimum") #interpolate only if a tracks' len greater than n_min
    parser.add_argument("--n_dti", type=int, default=20, help="dti") #max interpolate frames between two track entity
    parser.add_argument("--n_remove", type=int, default=0, help="dti") #if tracks' len is smaller than n_remove, the track will be removed
    parser.add_argument("--move_cond", type=str, default="{scale}", help="move condition") #if set even the frames between two track entity is large
    #than n_dti, as long as the move distance satisfy this option, we still performance interpolation

    return parser


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def write_results_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for i in range(results.shape[0]):
            frame_data = results[i]
            frame_id = int(frame_data[0])
            track_id = int(frame_data[1])
            x1, y1, w, h = frame_data[2:6]
            score = frame_data[6]
            line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
            f.write(line)


def dti(txt_path, save_path, n_min=25, n_dti=20,n_remove=0,move_cond=None):
    seq_txts = sorted(glob.glob(os.path.join(txt_path, '*.txt')))
    for seq_txt in seq_txts:
        seq_name = seq_txt.split('/')[-1]
        print(f"Process {seq_name}")
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=',')
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)
        for track_id in range(min_id, max_id + 1):
            index = (seq_data[:, 1] == track_id)
            tracklet = seq_data[index]
            tracklet_dti = tracklet
            if tracklet.shape[0] == 0:
                continue
            if tracklet.shape[0]<=n_remove:
                print(f"Remove {track_id}, track len = {tracklet.shape[0]}")
                continue
            n_frame = tracklet.shape[0]
            n_conf = np.sum(tracklet[:, 6] > 0.5)
            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                        (right_frame - left_frame) + left_bbox
                            frames_dti[curr_frame] = curr_bbox
                    elif right_frame - left_frame >= n_dti and move_cond is not None and len(move_cond)>1:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        bboxes = np.stack([right_bbox,left_bbox],axis=0)
                        scale = np.max(bboxes[:,2:])
                        right_cp = right_bbox[:2]+right_bbox[2:]/2
                        left_cp = left_bbox[:2]+left_bbox[2:]/2
                        dist_cp = np.linalg.norm(left_cp-right_cp)
                        dist_limit = float(move_cond.format(scale=scale))
                        if dist_cp<dist_limit:
                            print(f"dist limit is satisfy, right_frame={right_frame},left_frame={left_frame}, dist={dist_cp}")
                            for j in range(1, num_bi + 1):
                                curr_frame = j + left_frame
                                curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                            (right_frame - left_frame) + left_bbox
                                frames_dti[curr_frame] = curr_bbox

                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    print(f"Interpolation: {track_id}, frames = {data_dti[:,0].tolist()}, pos={data_dti[0,2:6]}")
                    tracklet_dti = np.vstack((tracklet, data_dti))
            seq_results = np.vstack((seq_results, tracklet_dti))
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        write_results_score(save_seq_txt, seq_results)


if __name__ == '__main__':
    args = make_parser().parse_args()

    if args.save_path is None:
        args.save_path = args.txt_path

    mkdir_if_missing(args.save_path)
    dti(args.txt_path, args.save_path, n_min=args.n_min, n_dti=args.n_dti,move_cond=args.move_cond)

