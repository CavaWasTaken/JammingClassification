#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Full sample of grabbing signals
# Author: Arman Ebrahimi Mehr
# Copyright: NavSAS
# GNU Radio version: 3.10.11.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from datetime import datetime
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import os
import osmosdr
import time
import sip
import threading



class dataset_generation(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Full sample of grabbing signals", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Full sample of grabbing signals")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "dataset_generation")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.time_stamp = time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.prefix_filename = prefix_filename = 'LINKS_L1_10Mhz_complex_int8_' + time_stamp
        self.jamming_type = jamming_type = 'CLEAN'
        self.jamming_power = jamming_power = 55
        self.root_dir = root_dir = '/home/arman-ebrahimi/Downloads/ATLANTIS_Dataset_20MHz'
        self.filename = filename = prefix_filename + '_' + jamming_type + '_' + str(jamming_power)
        self.samp_rate = samp_rate = 20e6
        self.rx_gain = rx_gain = 40
        self.full_filename = full_filename = os.path.join(root_dir, jamming_type, filename)
        self.f_L1 = f_L1 = 1575420000
        self.IQ_duration = IQ_duration = 180

        ##################################################
        # Blocks
        ##################################################

        self._rx_gain_range = qtgui.Range(0, 75, 1, 40, 200)
        self._rx_gain_win = qtgui.RangeWidget(self._rx_gain_range, self.set_rx_gain, "'rx_gain'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._rx_gain_win)
        self.qtgui_sink_x_0 = qtgui.sink_c(
            1024, #fftsize
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "", #name
            True, #plotfreq
            False, #plotwaterfall
            True, #plottime
            True, #plotconst
            None # parent
        )
        self.qtgui_sink_x_0.set_update_time(1.0/10)
        self._qtgui_sink_x_0_win = sip.wrapinstance(self.qtgui_sink_x_0.qwidget(), Qt.QWidget)

        self.qtgui_sink_x_0.enable_rf_freq(False)

        self.top_layout.addWidget(self._qtgui_sink_x_0_win)
        self.osmosdr_source_0 = osmosdr.source(
            args="numchan=" + str(1) + " " + "bladerf=c1ed7930fbad41d5bdc41cc810266af3"
        )
        self.osmosdr_source_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_source_0.set_sample_rate(samp_rate)
        self.osmosdr_source_0.set_center_freq(1575420000, 0)
        self.osmosdr_source_0.set_freq_corr(0, 0)
        self.osmosdr_source_0.set_dc_offset_mode(0, 0)
        self.osmosdr_source_0.set_iq_balance_mode(0, 0)
        self.osmosdr_source_0.set_gain_mode(True, 0)
        self.osmosdr_source_0.set_gain(0, 0)
        self.osmosdr_source_0.set_if_gain(0, 0)
        self.osmosdr_source_0.set_bb_gain(0, 0)
        self.osmosdr_source_0.set_antenna('', 0)
        self.osmosdr_source_0.set_bandwidth(samp_rate, 0)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, (int(samp_rate*IQ_duration)))
        self.blocks_file_sink_0_0_0 = blocks.file_sink(gr.sizeof_char*1, full_filename, False)
        self.blocks_file_sink_0_0_0.set_unbuffered(False)
        self.blocks_complex_to_interleaved_char_0 = blocks.complex_to_interleaved_char(False, 2**7)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_complex_to_interleaved_char_0, 0), (self.blocks_file_sink_0_0_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_complex_to_interleaved_char_0, 0))
        self.connect((self.osmosdr_source_0, 0), (self.blocks_head_0, 0))
        self.connect((self.osmosdr_source_0, 0), (self.qtgui_sink_x_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "dataset_generation")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_time_stamp(self):
        return self.time_stamp

    def set_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp
        self.set_prefix_filename('LINKS_L1_10Mhz_complex_int8_' + self.time_stamp)

    def get_prefix_filename(self):
        return self.prefix_filename

    def set_prefix_filename(self, prefix_filename):
        self.prefix_filename = prefix_filename
        self.set_filename(self.prefix_filename + '_' + self.jamming_type + '_' + str(self.jamming_power))

    def get_jamming_type(self):
        return self.jamming_type

    def set_jamming_type(self, jamming_type):
        self.jamming_type = jamming_type
        self.set_filename(self.prefix_filename + '_' + self.jamming_type + '_' + str(self.jamming_power))
        self.set_full_filename(os.path.join(self.root_dir, self.jamming_type, self.filename))

    def get_jamming_power(self):
        return self.jamming_power

    def set_jamming_power(self, jamming_power):
        self.jamming_power = jamming_power
        self.set_filename(self.prefix_filename + '_' + self.jamming_type + '_' + str(self.jamming_power))

    def get_root_dir(self):
        return self.root_dir

    def set_root_dir(self, root_dir):
        self.root_dir = root_dir
        self.set_full_filename(os.path.join(self.root_dir, self.jamming_type, self.filename))

    def get_filename(self):
        return self.filename

    def set_filename(self, filename):
        self.filename = filename
        self.set_full_filename(os.path.join(self.root_dir, self.jamming_type, self.filename))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_head_0.set_length((int(self.samp_rate*self.IQ_duration)))
        self.osmosdr_source_0.set_sample_rate(self.samp_rate)
        self.osmosdr_source_0.set_bandwidth(self.samp_rate, 0)
        self.qtgui_sink_x_0.set_frequency_range(0, self.samp_rate)

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain

    def get_full_filename(self):
        return self.full_filename

    def set_full_filename(self, full_filename):
        self.full_filename = full_filename
        self.blocks_file_sink_0_0_0.open(self.full_filename)

    def get_f_L1(self):
        return self.f_L1

    def set_f_L1(self, f_L1):
        self.f_L1 = f_L1

    def get_IQ_duration(self):
        return self.IQ_duration

    def set_IQ_duration(self, IQ_duration):
        self.IQ_duration = IQ_duration
        self.blocks_head_0.set_length((int(self.samp_rate*self.IQ_duration)))




def main(top_block_cls=dataset_generation, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
