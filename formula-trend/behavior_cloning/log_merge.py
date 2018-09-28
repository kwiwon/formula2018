#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################
#
# Merge several drive logs into one
#
######################################

import csv
import os
import os.path as path
import shutil

LOG_ROOT_DIR = "/change_this_to_your_log_folder_path"
TARGET_DIR = "_target"


def get_img_file_name(full_path):
    return path.split(full_path)[-1]


if __name__ == '__main__':

    csv_file_name = "driving_log.csv"
    img_dir = "IMG"

    # Get sub-folders
    root, dirs, files = next(os.walk(LOG_ROOT_DIR))

    # Filter out target folder
    dirs = sorted(filter(lambda s: not s.startswith(TARGET_DIR), dirs))

    # Create target folder
    os.makedirs(path.join(root, TARGET_DIR), exist_ok=True)
    os.makedirs(path.join(root, TARGET_DIR, img_dir), exist_ok=True)

    # Collect logs/images and copy to target folder
    with open(path.join(root, TARGET_DIR, csv_file_name), mode='w') as out:
        writer = csv.writer(out)
        count = 0

        # Traverse every log folder
        for log_dir in dirs:
            csv_file = path.join(root, log_dir, csv_file_name)
            if not path.exists(csv_file):
                continue

            print("Processing", csv_file)

            # Parse CSV file
            with open(csv_file, newline='') as f:
                for row in csv.reader(f):
                    img_name_src, rest = get_img_file_name(row[0]), row[1:]
                    img_name_dst = "{}_{}_{}".format(count, log_dir, img_name_src)

                    img_file = path.join(root, log_dir, img_dir, img_name_src)
                    if not path.exists(img_file):
                        continue

                    shutil.copy2(src=img_file,
                                 dst=path.join(root, TARGET_DIR, img_dir, img_name_dst))

                    rest.insert(0, img_name_dst)
                    writer.writerow(rest)
                    count += 1

    print("Processed", count, "record(s).")
