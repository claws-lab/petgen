#!/usr/bin/env bash
# The first 0 is job_id, the second 0 is gpu_id
# PETGEN is based on ICLR'19 RelGAN.
# we modify the relgan_instructor.py file and call the new revised class.
python run_relgan.py 0 0