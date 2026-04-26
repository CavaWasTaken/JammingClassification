#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Full sample of grabbing signals
# Author: Arman Ebrahimi Mehr
# Copyright: NavSAS
# GNU Radio version: 3.10.12.0

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import osmosdr
import time
import threading

# NUOVO sdr_base.py

class SdrBase(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Full sample of grabbing signals", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 10e6
        self.buffer_size = buffer_size = 500e-3
        self.num_samples_buffer = num_samples_buffer = int(buffer_size * samp_rate)
        self.f_L1 = f_L1 = 1575420000

        ##################################################
        # Blocks
        ##################################################

        self.osmosdr_source_0 = osmosdr.source(
            args="numchan=" + str(1) + " " + "hackrf=0"
        )
        self.osmosdr_source_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_source_0.set_sample_rate(samp_rate)
        self.osmosdr_source_0.set_center_freq(f_L1, 0)
        self.osmosdr_source_0.set_freq_corr(0, 0)
        self.osmosdr_source_0.set_dc_offset_mode(0, 0)
        self.osmosdr_source_0.set_iq_balance_mode(0, 0)
        self.osmosdr_source_0.set_gain_mode(False, 0)
        self.osmosdr_source_0.set_gain(0, 0)
        self.osmosdr_source_0.set_if_gain(16, 0)
        self.osmosdr_source_0.set_bb_gain(0, 0)
        self.osmosdr_source_0.set_antenna('', 0)
        self.osmosdr_source_0.set_bandwidth(samp_rate, 0)
        self.blocks_vector_sink_x_0 = blocks.vector_sink_c(1, 1024)
        self.blocks_correctiq_0 = blocks.correctiq()


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_correctiq_0, 0), (self.blocks_vector_sink_x_0, 0))
        self.connect((self.osmosdr_source_0, 0), (self.blocks_correctiq_0, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.osmosdr_source_0.set_sample_rate(self.samp_rate)
        self.osmosdr_source_0.set_bandwidth(self.samp_rate, 0)

    def get_buffer_size(self):
        return self.buffer_size
    
    def set_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_samples_buffer = int(self.buffer_size * self.samp_rate)

    def get_num_samples_buffer(self):
        return self.num_samples_buffer
    
    def set_num_samples_buffer(self, num_samples_buffer):
        self.num_samples_buffer = num_samples_buffer

    def get_f_L1(self):
        return self.f_L1

    def set_f_L1(self, f_L1):
        self.f_L1 = f_L1
        self.osmosdr_source_0.set_center_freq(self.f_L1, 0)




def main(top_block_cls=SdrBase, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
