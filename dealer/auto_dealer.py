#! /usr/bin/env python
######################################################################
#
#  auto_dealer.py - A ZMQ pub client that listens to the M&C system of
#  the GBT and responds to incoming values.
#
#  Copyright (C) 2015 Associated Universities, Inc. Washington DC, USA.
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#  General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
#
#  Correspondence concerning GBT software should be addressed as follows:
#  GBT Operations
#  National Radio Astronomy Observatory
#  P. O. Box 2
#  Green Bank, WV 24944-0002 USA
#
######################################################################

#import /users/rblack/bf/dibas/lib/python/dealer
import dealer
#from dealer_proxy import DealerProxy
from ZMQJSONProxy import ZMQJSONProxyException
#from /users/rblack/bf/dibas/lib/python/ZMQJSONProxy import ZMQJSONProxyException
import zmq
import sys

from PBDataDescriptor_pb2 import PBDataField
from DataStreamUtils import get_service_endpoints
from DataStreamUtils import get_every_parameter
from datetime import datetime, tzinfo, date, time
import pytz

auto_set = True
ctx = None
D = None
starttime = None


def state_callback(p):
    """This is called whenever scans are run on the GBT telescope. The GBT
    M&C system states transition to "Running" via "Comitted." This is
    thus a good state to look for to catch the system about to start a
    scan. Likewise, 'Stopping' or 'Aborting' is a a good indicator
    that a scan is coming to an end.

    """
    global D
    global starttime
    val = p.val_struct[0].val_string[0]

    if val == 'Activating':
       print "Activating..."
       if starttime.hour != 0 and starttime.minute != 0 and starttime.second != 0:
           print "   We're doing this at ", starttime
           D.start(starttime)
    if val == 'Running':
       print "Running..."
    if val == 'Committed':
       # D.set_status(OBS_MODE='RAW')
       #D.set_param(int_length=0.25)
       #D.startin(2,1400)
       print "Committed"
    elif val == "Stopping":
        #D.stop()
       print "Stopping"
    elif val == "Aborting":
       D.stop()
       print "Aborting"
    print datetime.now(), val

def start_time_callback(val):
    global D
    global starttime
    num_sec = val.val_struct[0].val_struct[0].val_double[0]

    date = datetime.utcnow().date()
    hour = int(num_sec/3600)
    minute = int((num_sec - hour*3600)/60)
    second = int(num_sec - hour*3600 - minute*60)
    my_time = time(hour, minute, second)
    starttime = datetime.combine(date, my_time)
    starttime = starttime.replace(tzinfo=pytz.utc)
    print starttime

def scan_len_callback(p):
    global D
    scan_length = p.val_struct[0].val_struct[0].val_double[0]
    print "scan length = ", scan_length
    D.set_param(scan_length = scan_length)


def main(url):
    global ctx, D, auto_set

    keys = {"ScanCoordinator.ScanCoordinator:P:state": state_callback,
          "ScanCoordinator.ScanCoordinator:P:startTime": start_time_callback,
          "ScanCoordinator.ScanCoordinator:P:scanLength": scan_len_callback}
    ctx = zmq.Context()
#    D = DealerProxy(ctx, url)
    D = dealer.Dealer()
    # D.add_active_player('BANKB', 'BANKC', 'BANKD', 'BANKE', 'BANKF')
    # Real ScanCoordinator
    req_url = "tcp://gbtdata.gbt.nrao.edu:5559"
    # Simulator ScanCoordinator
    # req_url = "tcp://toe.gb.nrao.edu:5559"
    subscriber = ctx.socket(zmq.SUB)
    D.set_mode('FLAG_CALCORR_MODE')
    #D.startin(2,10)
    #D.startin(5,5)

    for key in keys:
        print key
        major, minor = key.split(':')[0].split('.')
        print major, minor
        #params = get_every_parameter(ctx,major,minor)  
        #print params      
        sub_url, _, _ = get_service_endpoints(ctx, req_url, major, minor, 0)
        print sub_url
        subscriber.connect(sub_url)
        subscriber.setsockopt(zmq.SUBSCRIBE, key)
    while (auto_set):
        key, payload = subscriber.recv_multipart()
        df = PBDataField()
        df.ParseFromString(payload)
        f = keys[key]

        try:
            f(df)
        except ZMQJSONProxyException, e:
            print "Caught exception from Dealer", e
    subscriber.close()
    ctx.term


def signal_handler(signal, frame):
    global auto_set
    auto_set = False

if __name__ == '__main__':
    #if len(sys.argv) < 2:
    #    print "Need a URL to the Dealer daemon"
    #else:
    #    url = sys.argv[1]
    #    main(url)
    main('')
