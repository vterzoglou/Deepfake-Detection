import csv
import subprocess
from ffmpeg import probe
import numpy as np


def compress_files(sources, labels,  crf):
    """
    Function that takes as input a list of paths to source uncompressed (c0) videos
    and compresses them to a quality specified by the crf factor.

    :param sources: list of c0 (source) videos paths
    :param crf: Constant Rate Factor to be used in compressing videos
    :return: None
    """
    target_names = []
    target_n_bitrates = []
    for video_file in sources:
        # Source paths are supposed to contain the substring "c0"
        # which is replaced to produce target paths
        target_name = video_file.replace("c0","c"+str(crf))

        # If executed through a shell, consider adding shell=True and stdout parameters
        process = subprocess.run(f'ffmpeg -r 30 -i {video_file} -crf {crf} -c:v libx264'
                                 f' {target_name}'
                                 f' -hide_banner -loglevel error')

        target_names.append([target_name])

        # Bitrate metadata are included in the index file
        infos = probe(target_name)
        video_bitrate = int(infos['streams'][0]['bit_rate'])
        video_width = infos['streams'][0]['width']
        video_height = infos['streams'][0]['height']
        video_norm_bitrate = video_bitrate / (video_width * video_height)
        target_n_bitrates.append(video_norm_bitrate)

    target_n_bitrates = np.array(target_n_bitrates)
    target_log_n_brs = np.log2(target_n_bitrates)

    target_index_file = f'./indexes/c{CRF}_video_index.csv'
    with open(target_index_file, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        for name, label, nbr, lnbr in zip(target_names, labels, target_n_bitrates, target_log_n_brs):
            w.writerow([name, label, nbr, lnbr])
    return


if __name__ == '__main__':

    CRF = 31

    source_index_file = './indexes/c0_video_index.csv'
    sources = []
    labels = []
    with open(source_index_file, 'r', newline='', encoding='utf-8-sig') as f:
        r = csv.reader(f)
        for row in r:
            sources.append(row[0])
            labels.append(row[1])
    compress_files(sources, labels, crf=CRF)
