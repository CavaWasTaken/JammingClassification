#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: jammer_tx_OSMO
# Author: Arman Ebrahimi Mehr
# Copyright: NavSAS
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import jammer_tx_OSMO_epy_block_0 as epy_block_0  # embedded python block
import osmosdr
import time
import sip



class jammer_tx_OSMO(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "jammer_tx_OSMO", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("jammer_tx_OSMO")
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

        self.settings = Qt.QSettings("GNU Radio", "jammer_tx_OSMO")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.freq_L1 = freq_L1 = 1.57542e9
        self.fc_variate = fc_variate = 0
        self.tx_gain_period = tx_gain_period = 120000
        self.tx_gain = tx_gain = 1
        self.samp_rate = samp_rate = 20e6
        self.interference_fc = interference_fc = freq_L1 + fc_variate
        self.bin_file = bin_file = 'interference_bin/triangular-wave_5_20.bin'

        ##################################################
        # Blocks
        ##################################################

        self._tx_gain_range = qtgui.Range(0, 50, 1, 1, 200)
        self._tx_gain_win = qtgui.RangeWidget(self._tx_gain_range, self.set_tx_gain, "tx_gain", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._tx_gain_win)
        # Create the options list
        self._bin_file_options = ['interference_bin/linear_narrow_10_20.bin', 'interference_bin/linear_wide_fast_20.bin', 'interference_bin/tick_3_20.bin', 'interference_bin/triangular-wave_5_20.bin', 'interference_bin/triangular_15_20.bin']
        # Create the labels list
        self._bin_file_labels = ['Linear Narrow', 'Linear Wide Fast', 'Tick', 'Triangular Wave', 'Triangular']
        # Create the combo box
        self._bin_file_tool_bar = Qt.QToolBar(self)
        self._bin_file_tool_bar.addWidget(Qt.QLabel("Interferent" + ": "))
        self._bin_file_combo_box = Qt.QComboBox()
        self._bin_file_tool_bar.addWidget(self._bin_file_combo_box)
        for _label in self._bin_file_labels: self._bin_file_combo_box.addItem(_label)
        self._bin_file_callback = lambda i: Qt.QMetaObject.invokeMethod(self._bin_file_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._bin_file_options.index(i)))
        self._bin_file_callback(self.bin_file)
        self._bin_file_combo_box.currentIndexChanged.connect(
            lambda i: self.set_bin_file(self._bin_file_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._bin_file_tool_bar)
        self.qtgui_sink_x_0 = qtgui.sink_c(
            1024, #fftsize
            window.WIN_BLACKMAN_hARRIS, #wintype
            interference_fc, #fc
            samp_rate, #bw
            "", #name
            True, #plotfreq
            True, #plotwaterfall
            True, #plottime
            True, #plotconst
            None # parent
        )
        self.qtgui_sink_x_0.set_update_time(1.0/10)
        self._qtgui_sink_x_0_win = sip.wrapinstance(self.qtgui_sink_x_0.qwidget(), Qt.QWidget)

        self.qtgui_sink_x_0.enable_rf_freq(True)

        self.top_layout.addWidget(self._qtgui_sink_x_0_win)
        self.osmosdr_sink_0_0 = osmosdr.sink(
            args="numchan=" + str(1) + " " + "hackrf=0"
        )
        self.osmosdr_sink_0_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_sink_0_0.set_sample_rate(samp_rate)
        self.osmosdr_sink_0_0.set_center_freq(interference_fc, 0)
        self.osmosdr_sink_0_0.set_freq_corr(0, 0)
        self.osmosdr_sink_0_0.set_gain(16, 0)
        self.osmosdr_sink_0_0.set_if_gain(tx_gain, 0)
        self.osmosdr_sink_0_0.set_bb_gain(0, 0)
        self.osmosdr_sink_0_0.set_antenna('', 0)
        self.osmosdr_sink_0_0.set_bandwidth(0, 0)
        self._fc_variate_range = qtgui.Range(-10e6, +10e6, 1e6, 0, 200)
        self._fc_variate_win = qtgui.RangeWidget(self._fc_variate_range, self.set_fc_variate, "fc_variate", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._fc_variate_win)
        self.epy_block_0 = epy_block_0.blk(start_gain=0, step=5, max_gain=40)
        self.blocks_message_strobe_0 = blocks.message_strobe(pmt.intern("tick"), tx_gain_period)
        self.blocks_file_source_0_1_0 = blocks.file_source(gr.sizeof_gr_complex*1, bin_file, True, 0, 0)
        self.blocks_file_source_0_1_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_0_1_0.set_block_alias("LAMPEDUSA")


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0, 'strobe'), (self.epy_block_0, 'in'))
        self.connect((self.blocks_file_source_0_1_0, 0), (self.osmosdr_sink_0_0, 0))
        self.connect((self.blocks_file_source_0_1_0, 0), (self.qtgui_sink_x_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "jammer_tx_OSMO")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_freq_L1(self):
        return self.freq_L1

    def set_freq_L1(self, freq_L1):
        self.freq_L1 = freq_L1
        self.set_interference_fc(self.freq_L1 + self.fc_variate)

    def get_fc_variate(self):
        return self.fc_variate

    def set_fc_variate(self, fc_variate):
        self.fc_variate = fc_variate
        self.set_interference_fc(self.freq_L1 + self.fc_variate)

    def get_tx_gain_period(self):
        return self.tx_gain_period

    def set_tx_gain_period(self, tx_gain_period):
        self.tx_gain_period = tx_gain_period
        self.blocks_message_strobe_0.set_period(self.tx_gain_period)

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.osmosdr_sink_0_0.set_if_gain(self.tx_gain, 0)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.osmosdr_sink_0_0.set_sample_rate(self.samp_rate)
        self.qtgui_sink_x_0.set_frequency_range(self.interference_fc, self.samp_rate)

    def get_interference_fc(self):
        return self.interference_fc

    def set_interference_fc(self, interference_fc):
        self.interference_fc = interference_fc
        self.osmosdr_sink_0_0.set_center_freq(self.interference_fc, 0)
        self.qtgui_sink_x_0.set_frequency_range(self.interference_fc, self.samp_rate)

    def get_bin_file(self):
        return self.bin_file

    def set_bin_file(self, bin_file):
        self.bin_file = bin_file
        self._bin_file_callback(self.bin_file)
        self.blocks_file_source_0_1_0.open(self.bin_file, True)




def main(top_block_cls=jammer_tx_OSMO, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

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
