# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:54:08 2020

@author: Jin Dou
"""
import mne
import numpy as np

sfreq = 1000
times = np.arange(0, 10, 0.001)  # Use 10000 samples (10s)

sin = np.sin(times * 10)  # Multiplied by 10 for shorter cycles
cos = np.cos(times * 10)
event_id = 1  # This is used to identify the events.
# First column is for the sample number.
events = np.array([[200, 0, event_id],
                   [1200, 0, event_id],
                   [2000, 0, event_id]])  # List of three arbitrary events

# Here a data set of 700 ms epochs from 2 channels is
# created from sin and cos data.
# Any data in shape (n_epochs, n_channels, n_times) can be used.
epochs_data = np.array([[sin[:700], cos[:700]],
                        [sin[1000:1700], cos[1000:1700]],
                        [sin[1800:2500], cos[1800:2500]]])

ch_names = ['sin', 'cos']
ch_types = ['mag', 'mag']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

epochs = mne.EpochsArray(epochs_data, info=info, events=events,
                         event_id={'arbitrary': 1,'aa':2})

picks = mne.pick_types(info, meg=True, eeg=False, misc=False)

epochs.plot(picks=picks, scalings='auto', show=True, block=True)