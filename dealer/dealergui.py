import wx
import wx.lib.colourdb
import dealer

class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        # Inherit from Frame
        wx.Frame.__init__(self, parent, title=title)

        # Update color database so we have access to a lot more colors
        wx.lib.colourdb.updateColourDB()

        # Create timer for periodic updates
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)

        # Add a panel to make things look good cross-platform
        self.panel = wx.Panel(self, wx.ID_ANY)

        # Set up masterSizer
        masterSizer = wx.BoxSizer(wx.VERTICAL)
        # Set up titleSizer
        titleSizer = wx.BoxSizer(wx.HORIZONTAL)
        # Set up contentSizer
        contentSizer = wx.BoxSizer(wx.HORIZONTAL)
        # Set up configSizer
        configSizer = wx.BoxSizer(wx.VERTICAL)
        # Set up statusSizer
        statusSizer = wx.BoxSizer(wx.VERTICAL)
        # Set up buttonSizer
        buttonSizer = wx.BoxSizer(wx.VERTICAL)

	# Add spacer before title
        masterSizer.AddSpacer((-1, 10))

        # Create title
        titleLabel = wx.StaticText(self.panel, wx.ID_ANY, "FLAG Control Interface")
        titleSizer.Add(titleLabel, 0, wx.ALL)
        masterSizer.Add(titleSizer, 0, wx.CENTER)

        # Add space after title
        masterSizer.AddSpacer((-1, 10))


        # Add headers for player information 
        playerText = wx.StaticText(self.panel, wx.ID_ANY, "Player")
        xmqText = wx.StaticText(self.panel, wx.ID_ANY, "Connection Details")
        headerSizer = wx.BoxSizer(wx.HORIZONTAL)
        headerSizer.Add(playerText, 0, wx.LEFT|wx.ALIGN_CENTER)
        headerSizer.AddSpacer((50, -1))
        headerSizer.Add(xmqText, 0, wx.EXPAND)
        configSizer.Add(headerSizer, 0, wx.LEFT)
        configSizer.AddSpacer((-1, 13))

        # Populate player information
        players = d.list_available_players()
        self.portInfo = dict()

        # Check if players are active to get their IP/Port information
        for playerName in sorted(players):
            if playerName in d.list_active_players():
                info = d.players[playerName].url
            else:
                info = "INACTIVE"

            # Create activation toggle button for each player
            infoLabel = wx.Button(self.panel, wx.ID_ANY, playerName)

            # Bind buttons to toggleActivate
            self.Bind(wx.EVT_BUTTON, self.toggleActivate, infoLabel)

            # Create text box with IP/Port information
            self.portInfo[playerName] = wx.TextCtrl(self.panel, wx.ID_ANY, info, size=(125,-1), style=wx.TE_READONLY)
            self.portInfo[playerName].SetBackgroundColour(wx.NamedColour("dark grey"))

            # Create small sizer to organize buttons/text boxes per player
            infoSizer = wx.BoxSizer(wx.HORIZONTAL)
            infoSizer.Add(infoLabel, 0, wx.LEFT|wx.ALIGN_CENTER)
            infoSizer.AddSpacer((5, -1))
            infoSizer.Add(self.portInfo[playerName], 0, wx.EXPAND)
            configSizer.Add(infoSizer, 0, wx.LEFT)
            configSizer.AddSpacer((-1, 5))

        # Create sizer for activate/deactivate all buttons
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)
        actButton = wx.Button(self.panel, wx.ID_ANY, "Activate All")
        self.Bind(wx.EVT_BUTTON, self.activateAll, actButton)
        btnSizer.Add(actButton, 0, wx.LEFT)
        deactButton = wx.Button(self.panel, wx.ID_ANY, "Deactivate All")
        self.Bind(wx.EVT_BUTTON, self.deactivateAll, deactButton)
        btnSizer.Add(deactButton, 0, wx.RIGHT)
        configSizer.Add(btnSizer, 0, wx.EXPAND)

        # Add sizer to parent sizer
        contentSizer.AddSpacer((10,-1))
        contentSizer.Add(configSizer, 0, wx.EXPAND)
           


        # Populate status information
        activePlayers = d.list_active_players()
        
        # Create horizontal box with text boxes to specify desired shared memory
        specSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.statusNames = []
        self.statusNames.append(wx.TextCtrl(self.panel, wx.ID_ANY, "BINDHOST", style=wx.TE_CENTRE, size=(90,-1)))
        self.statusNames.append(wx.TextCtrl(self.panel, wx.ID_ANY, "BINDPORT", style=wx.TE_CENTRE, size=(90,-1)))
        self.statusNames.append(wx.TextCtrl(self.panel, wx.ID_ANY, "XID", style=wx.TE_CENTRE, size=(90,-1)))
        self.statusNames.append(wx.TextCtrl(self.panel, wx.ID_ANY, "NETMCNT", style=wx.TE_CENTRE, size=(90,-1)))
        self.statusNames.append(wx.TextCtrl(self.panel, wx.ID_ANY, "NETSTAT", style=wx.TE_CENTRE, size=(90,-1)))
        self.statusNames.append(wx.TextCtrl(self.panel, wx.ID_ANY, "CORSTAT", style=wx.TE_CENTRE, size=(90,-1)))

        # Add text boxes to sizer
        for i in range(0,6):
            specSizer.Add(self.statusNames[i], 0, wx.LEFT)
            specSizer.AddSpacer((5, -1))
        statusSizer.Add(specSizer, 0, wx.LEFT)
        statusSizer.AddSpacer((-1, 5))
        
        # Create status data structure
        self.statusBox = []
        for i in range(0,6):
            self.statusBox.append(dict())

        for playerName in sorted(players):
            # Create horizontal box sizer for information
            bankSizer = wx.BoxSizer(wx.HORIZONTAL)
            for i in range(0,6):
                self.statusBox[i][playerName] = wx.TextCtrl(self.panel, wx.ID_ANY, "", style=wx.TE_READONLY, size=(90,-1))
                bankSizer.Add(self.statusBox[i][playerName], 0, wx.LEFT)
                bankSizer.AddSpacer((5, -1))
            statusSizer.Add(bankSizer, 0, wx.EXPAND)
            statusSizer.AddSpacer((-1, 5))

        # Add mode selection controls
        modeSizer = wx.BoxSizer(wx.HORIZONTAL)
        modes = d.list_modes()
        self.modeBox = wx.ComboBox(self.panel, wx.ID_ANY, choices=modes)
        self.modeBox.SetValue(modes[0])
        modeSizer.Add(self.modeBox, 0, wx.LEFT)
        modeButton = wx.Button(self.panel, wx.ID_ANY, "Set Mode")
        self.Bind(wx.EVT_BUTTON, self.setMode, modeButton)
        modeSizer.Add(modeButton, 0, wx.RIGHT)
        statusSizer.Add(modeSizer, 0, wx.EXPAND)

        # Add sizer to parent sizer
        contentSizer.AddSpacer((10,-1))
        contentSizer.Add(statusSizer, 0, wx.EXPAND)
        
        # Populate buttons
        start = wx.Button(self.panel, label="START")
        start.SetBackgroundColour('GREEN')
        self.Bind(wx.EVT_BUTTON, self.sendStart, start)
        stop = wx.Button(self.panel, label="STOP")
        stop.SetBackgroundColour('RED')
        self.Bind(wx.EVT_BUTTON, self.sendStop, stop)
        buttonSizer.Add(start, 0, wx.EXPAND)
        buttonSizer.AddSpacer((-1, 5))
        buttonSizer.Add(stop, 0, wx.EXPAND)
        startText = wx.StaticText(self.panel, wx.ID_ANY, "Start Delay (sec)")
        self.scanStart = wx.TextCtrl(self.panel, wx.ID_ANY, "5")
        lengthText = wx.StaticText(self.panel, wx.ID_ANY, "Scan Length (sec)")
        self.scanLength = wx.TextCtrl(self.panel, wx.ID_ANY, "1")
        intText = wx.StaticText(self.panel, wx.ID_ANY, "Integration Length (sec)")
        self.intLength = wx.TextCtrl(self.panel, wx.ID_ANY, "1")

        buttonSizer.Add(startText, 0, wx.EXPAND);
        buttonSizer.Add(self.scanStart, 0, wx.EXPAND);
        buttonSizer.Add(lengthText, 0, wx.EXPAND);
        buttonSizer.Add(self.scanLength, 0, wx.EXPAND);
        buttonSizer.Add(intText, 0, wx.EXPAND);
        buttonSizer.Add(self.intLength, 0, wx.EXPAND);
        
        contentSizer.AddSpacer((10,-1))
        contentSizer.Add(buttonSizer, 0, wx.EXPAND)
        contentSizer.AddSpacer((10,-1))
        masterSizer.Add(contentSizer, 0, wx.EXPAND)

        self.panel.SetSizer(masterSizer)
        masterSizer.Fit(self)
        self.Show(True)
        self.timer.Start(250)

    def toggleActivate(self, event):
        playerName = event.GetEventObject().GetLabel()
        if playerName in d.list_active_players():
            d.remove_active_player(playerName)
        else:
            d.add_active_player(playerName)

    def activateAll(self, event):
        for playerName in d.list_available_players():
            d.add_active_player(playerName)

    def deactivateAll(self, event):
        for playerName in d.list_available_players():
            d.remove_active_player(playerName)

    def setMode(self, event):
        mode_name = self.modeBox.GetValue()
        d.set_mode(mode_name)

    def sendStart(self, event):
        startTime = float(self.scanStart.GetValue());
        scanLength = float(self.scanLength.GetValue());
        intLength = float(self.intLength.GetValue());
        d.set_param(int_length=intLength)
        d.startin(startTime,scanLength)

    def sendStop(self, event):
        d.stop()

    def update(self, event):
        for playerName in d.list_available_players():
            if playerName in d.list_active_players():
                info = d.players[playerName].url
            else:
                info = "INACTIVE"
            self.portInfo[playerName].SetValue(info)
            if playerName in d.list_active_players():
                if (d.players[playerName]._initialized):
                    if (d.players[playerName].get_mode() != None):
                        status_mem = d.players[playerName].get_status()
                        for i in range(0,6):
                            try:
                                self.statusBox[i][playerName].SetValue(str(status_mem[self.statusNames[i].GetValue()]))
                            except Exception, e:
                                self.statusBox[i][playerName].SetValue("N/A")
                                pass
        

d = dealer.Dealer()
app = wx.App(False)
frame = MyFrame(None, "FLAG Control Interface")
app.MainLoop()
