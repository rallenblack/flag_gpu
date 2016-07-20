function varargout = total_power_gui(varargin)
% TOTAL_POWER_GUI MATLAB code for total_power_gui.fig
%      TOTAL_POWER_GUI, by itself, creates a new TOTAL_POWER_GUI or raises the existing
%      singleton*.
%
%      H = TOTAL_POWER_GUI returns the handle to a new TOTAL_POWER_GUI or the handle to
%      the existing singleton*.
%
%      TOTAL_POWER_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TOTAL_POWER_GUI.M with the given input arguments.
%
%      TOTAL_POWER_GUI('Property','Value',...) creates a new TOTAL_POWER_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before total_power_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to total_power_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help total_power_gui

% Last Modified by GUIDE v2.5 15-Jul-2016 15:11:30

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @total_power_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @total_power_gui_OutputFcn, ...
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


% --- Executes just before total_power_gui is made visible.
function total_power_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to total_power_gui (see VARARGIN)

% Choose default command line output for total_power_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% Populate axes with NaN
axes(handles.ax_blade1);
plot(1:100, zeros(100,1));
title('ROACH 1');
xlabel('Time Samples');
ylabel('Power (arb. units)');
axes(handles.ax_blade2);
plot(1:100, zeros(100,1));
title('ROACH 2');
xlabel('Time Samples');
ylabel('Power (arb. units)');
axes(handles.ax_blade3);
plot(1:100, zeros(100,1));
title('ROACH 3');
xlabel('Time Samples');
ylabel('Power (arb. units)');
axes(handles.ax_blade4);
plot(1:100, zeros(100,1));
title('ROACH 4');
xlabel('Time Samples');
ylabel('Power (arb. units)');
axes(handles.ax_blade5);
plot(1:100, zeros(100,1));
title('ROACH 5');
xlabel('Time Samples');
ylabel('Power (arb. units)');

% Create timer callback
handles.timer = timer(...
        'ExecutionMode', 'fixedRate',...
        'Period', '0.01',...
        'TimerFcn', {@update_display,handles});

% Populate filename textedit
datadir = get(handles.edit_datadir, 'String');
set(handles.edit_filename, 'String', [datadir, '/power_0_mcnt_0.out']);






% UIWAIT makes total_power_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = total_power_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object creation, after setting all properties.
function ax_blade1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ax_blade1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate ax_blade1


% --- Executes on button press in togglebutton1.
function togglebutton1_Callback(hObject, eventdata, handles)
% hObject    handle to togglebutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglebutton1
button_pressed = get(hObject, 'Value');
if button_pressed
    datadir    = get(handles.edit_datadir, 'String');
    mcnt_step  = str2double(get(handles.edit_mcnt_step, 'String'));
    Ninstances = str2double(get(handles.edit_Ninstances, 'String'));
    Nant       = str2double(get(handles.edit_Nant, 'String'));
    if strcmp(get(handles.timer, 'Running'), 'off')
        start(handles.timer);
    end
else
    if strcmp(get(handles.timer, 'Running'), 'on')
        stop(handles.timer);
    end
end

% --- Executes when timer expires.
function update_display(handles)
    disp('Hello World!')


function edit_datadir_Callback(hObject, eventdata, handles)
% hObject    handle to edit_datadir (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_datadir as text
%        str2double(get(hObject,'String')) returns contents of edit_datadir as a double


% --- Executes during object creation, after setting all properties.
function edit_datadir_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_datadir (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_mcnt_step_Callback(hObject, eventdata, handles)
% hObject    handle to edit_mcnt_step (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_mcnt_step as text
%        str2double(get(hObject,'String')) returns contents of edit_mcnt_step as a double


% --- Executes during object creation, after setting all properties.
function edit_mcnt_step_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_mcnt_step (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_Ninstances_Callback(hObject, eventdata, handles)
% hObject    handle to edit_Ninstances (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_Ninstances as text
%        str2double(get(hObject,'String')) returns contents of edit_Ninstances as a double


% --- Executes during object creation, after setting all properties.
function edit_Ninstances_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_Ninstances (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_Nant_Callback(hObject, eventdata, handles)
% hObject    handle to edit_Nant (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_Nant as text
%        str2double(get(hObject,'String')) returns contents of edit_Nant as a double


% --- Executes during object creation, after setting all properties.
function edit_Nant_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_Nant (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_filename_Callback(hObject, eventdata, handles)
% hObject    handle to edit_filename (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_filename as text
%        str2double(get(hObject,'String')) returns contents of edit_filename as a double


% --- Executes during object creation, after setting all properties.
function edit_filename_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_filename (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
if strcmp(get(handles.timer, 'Running'), 'on')
    stop(handles.timer);
end
delete(handles.timer);

delete(hObject);
