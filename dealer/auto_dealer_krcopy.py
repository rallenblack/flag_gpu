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

import dealer
#from dealer_proxy import DealerProxy
from ZMQJSONProxy import ZMQJSONProxyException
import zmq
import sys

from PBDataDescriptor_pb2 import PBDataField
from DataStreamUtils import get_service_endpoints
from DataStreamUtils import get_every_parameter
from datetime import datetime

auto_set = True
ctx = None
D = None


def state_callback(p):
    """This is called whenever scans are run on the GBT telescope. The GBT
    M&C system states transition to "Running" via "Comitted." This is
    thus a good state to look for to catch the system about to start a
    scan. Likewise, 'Stopping' or 'Aborting' is a a good indicator
    that a scan is coming to an end.

    """
    global D
    val = p.val_struct[0].val_string[0]

    if val == 'Committed':
       # D.set_status(OBS_MODE='RAW')
       # D.set_param(int_length=0.25)
       # D.startin(15,120)
       print "Committed"
    elif val == "Stopping":
        #D.stop()
       print "Stopping"
    elif val == "Aborting":
        #D.stop()
       print "Aborting"
    print datetime.now(), val

def start_time_callback(p):
    print "Start time callback!"
    print "p.vstruct0:", p.val_struct[0].val_int64

def main(url):
    global ctx, D, auto_set

    keys = {"ScanCoordinator.ScanCoordinator:P:state": state_callback,
           "ScanCoordinator.ScanCoordinator:P:startTime": start_time_callback}
    ctx = zmq.Context()
#    D = DealerProxy(ctx, url)
    #D = dealer.Dealer()
    # Real ScanCoordinator
    req_url = "tcp://gbtdata.gbt.nrao.edu:5559"
    # Simulator ScanCoordinator
    #req_url = "tcp://toe.gb.nrao.edu:5559"
    subscriber = ctx.socket(zmq.SUB)
    #D.set_mode('FLAG_CALCORR_MODE')


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
