import pmt
from gnuradio import gr

class blk(gr.basic_block):
    def __init__(self, start_gain=0.0, step=1.0, max_gain=40.0):
        gr.basic_block.__init__(
            self,
            name="tx_gain_controller",
            in_sig=None,
            out_sig=None,
        )

        # Parameters
        self.gain = float(start_gain)
        self.step = float(step)
        self.max_gain = float(max_gain)

        # Message ports
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self.handle_msg)

        self.message_port_register_out(pmt.intern("out"))

        # Send initial gain command at start (optional)
        self._send_gain_cmd()

    def _send_gain_cmd(self):
        # Command dictionary for osmosdr: { 'gain': value }
        cmd = pmt.to_pmt({'gain': float(self.gain)})
        self.message_port_pub(pmt.intern("out"), cmd)

    def handle_msg(self, msg):
        # Triggered every 2 minutes by Message Strobe
        if self.gain < self.max_gain:
            self.gain = min(self.gain + self.step, self.max_gain)
            self._send_gain_cmd()
            print(self.gain)
        else:
            # Already at max, do nothing (no more increase)
            pass

