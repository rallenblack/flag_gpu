function varargout = beamformer_disp_gui(varargin)
% BEAMFORMER_DISP_GUI MATLAB code for beamformer_disp_gui.fig
%      BEAMFORMER_DISP_GUI, by itself, creates a new BEAMFORMER_DISP_GUI or raises the existing
%      singleton*.
%
%      H = BEAMFORMER_DISP_GUI returns the handle to a new BEAMFORMER_DISP_GUI or the handle to
%      the existing singleton*.
%
%      BEAMFORMER_DISP_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BEAMFORMER_DISP_GUI.M with the given input arguments.
%
%      BEAMFORMER_DISP_GUI('Property','Value',...) creates a new BEAMFORMER_DISP_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before beamformer_disp_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to beamformer_disp_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help beamformer_disp_gui

% Last Modified by GUIDE v2.5 15-Jul-2016 14:53:54

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @beamformer_disp_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @beamformer_disp_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before beamformer_disp_gui is made visible.
function beamformer_disp_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to beamformer_disp_gui (see VARARGIN)

% Choose default command line output for beamformer_disp_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

beamformer_display; % Call beamformer_display and only run gui, otherwise comment out "axes()"

% UIWAIT makes beamformer_disp_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = beamformer_disp_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object creation, after setting all properties.
function axes1_CreateFcn(hObject, eventdata, handles)
    figure();
        imagesc(1:N_beam1, 1:N_bin, 10*log10(abs(squeeze(power_acc_x))));

% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes1
