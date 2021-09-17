import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
from general_spectra_plotter import plot_spectra,availible_colors
import sys
import os
from argparse import ArgumentParser

def Run_GUI():
    color = '#d5ceda'
    text_color = 'black'
    availible_colors.sort()

    #Functions to prevent GUI blurring
    def make_dpi_aware():
        import ctypes
        import platform
        if int(platform.release()) >= 8:
            ctypes.windll.shcore.SetProcessDpiAwareness(True)
    #make_dpi_aware()

    #Function for drawing
    def draw_figure(canvas, figure):
    #    figure.tight_layout()
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both')#, expand=1)
        return figure_canvas_agg

    def resource_path(relative_path):
        base_path = getattr(
                sys,
                'MEIPASS',
                os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    image = resource_path('plot_spectra.png')
    image_icon = resource_path('plot_spectra.ico')
    #print(image_icon)

    left_pannel = sg.Column( [

                    [sg.Frame('Mol1',[
                            [sg.Text('Name:',justification='left',background_color=color,text_color=text_color),sg.InputText(key='Mol1_name',size=(54,1))],
                            [sg.Text('Excitations:',background_color=color,text_color=text_color),sg.Input(visible=False,enable_events=True,key='Mol1_inputs'),sg.FilesBrowse(),
                            sg.Text('Equilibrium Excitations:',background_color=color,text_color=text_color),sg.Input(visible=False,enable_events=True,key='Mol1_equilibrium'),sg.FileBrowse()],
                            [sg.Text('Color:',background_color=color,text_color=text_color),sg.Combo(availible_colors,size=(27,1),key='Mol1_color',bind_return_key=True)]
                            ],background_color=color,title_color=text_color
    )],

                    [sg.Frame('Mol2',[
                            [sg.Text('Name:',background_color=color,text_color=text_color),sg.InputText(key='Mol2_name',size=(54,1))],
                            [sg.Text('Excitations:',background_color=color,text_color=text_color),sg.Input(visible=False,enable_events=True,key='Mol2_inputs'),sg.FilesBrowse(),
                            sg.Text('Equilibrium Excitations:',background_color=color,text_color=text_color),sg.Input(visible=False,enable_events=True,key='Mol2_equilibrium'),sg.FileBrowse()],
                            [sg.Text('Color:',background_color=color,text_color=text_color),sg.Combo(availible_colors,size=(27,1),key='Mol2_color',bind_return_key=True)]

                            ],background_color=color,title_color=text_color)],

                    [sg.Frame('Mol3',[
                            [sg.Text('Name:',background_color=color,text_color=text_color),sg.InputText(key='Mol3_name',size=(54,1))],
                            [sg.Text('Excitations:',background_color=color,text_color=text_color),sg.Input(visible=False,enable_events=True,key='Mol3_inputs'),sg.FilesBrowse(),
                            sg.Text('Equilibrium Excitations:',background_color=color,text_color=text_color),sg.Input(visible=False,enable_events=True,key='Mol3_equilibrium'),sg.FileBrowse()],
                            [sg.Text('Color:',background_color=color,text_color=text_color),sg.Combo(availible_colors,size=(27,1),key='Mol3_color',bind_return_key=True)]

                            ],background_color=color,title_color=text_color)],


                    [sg.Frame('Mol4',[
                            [sg.Text('Name:',background_color=color,text_color=text_color),sg.InputText(key='Mol4_name',size=(54,1))],
                            [sg.Text('Excitations:',background_color=color,text_color=text_color),sg.Input(visible=False,enable_events=True,key='Mol4_inputs'),sg.FilesBrowse(),
                            sg.Text('Equilibrium Excitations:',background_color=color,text_color=text_color),sg.Input(visible=False,enable_events=True,key='Mol4_equilibrium'),sg.FileBrowse()],
                            [sg.Text('Color:',background_color=color,text_color=text_color),sg.Combo(availible_colors,size=(27,1),key='Mol4_color',bind_return_key=True)]

                            ],background_color=color,title_color=text_color)],



                    [sg.Frame('Mol5',[
                            [sg.Text('Name:',background_color=color,text_color=text_color),sg.InputText(key='Mol5_name',size=(54,1))],
                            [sg.Text('Excitations:',background_color=color,text_color=text_color),sg.Input(visible=False,enable_events=True,key='Mol5_inputs'),sg.FilesBrowse(),
                            sg.Text('Equilibrium Excitations:',background_color=color,text_color=text_color),sg.Input(visible=False,enable_events=True,key='Mol5_equilibrium'),sg.FileBrowse()],
                            [sg.Text('Color:',background_color=color,text_color=text_color),sg.Combo(availible_colors,size=(27,1),key='Mol5_color',bind_return_key=True)]

                            ],background_color=color,title_color=text_color)],


                            ],justification='left',background_color=color)

    right_pannel = sg.Column([[

        sg.Canvas(key='-CANVAS-')


        ]],background_color=color)


    top_menu = sg.Column([[sg.Image(image,size=(100,100),background_color=color),sg.Column([[sg.Frame('Plotting Controls',[
                        [
                        sg.Column([[sg.Text('Show Sampling',background_color=color,text_color=text_color)],[sg.Checkbox('',key='show_sampling',background_color=color,text_color=text_color)]],background_color=color),
                        sg.Column([[sg.Text('sigma',background_color=color,text_color=text_color)],[sg.InputText(0.25,size=(7,1),key='sigma')]],background_color=color),

                        sg.Column([[sg.Text('Start (nm)',background_color=color,text_color=text_color)],[sg.InputText(200,size=(7,1),key='start')]],background_color=color),
                        sg.Column([[sg.Text('End (nm)',background_color=color,text_color=text_color)],[sg.InputText(600,size=(7,1),key='stop')]],background_color=color),
                        sg.Column([[sg.Text('Step (nm)',background_color=color,text_color=text_color)],[sg.InputText(0.2,size=(7,1),key='step')]],background_color=color),

                        sg.Column([[sg.Text('Image File Name',background_color=color,text_color=text_color)],[sg.InputText(key='image_file')]],background_color=color),
                        sg.Column([[sg.Button('PLOT',key='plot')]],background_color=color),
                        sg.Column([[sg.Button('SAVE',key='save')]],background_color=color),
                        sg.Column([[sg.Button('Help/Info',key='help_info')]],background_color=color),
                        sg.Column([[sg.Button('EXIT',key='Exit')]],background_color=color)

                        ]

                        ],background_color=color,title_color=text_color)]],justification='center',background_color=color)]] ,background_color=color)

    bottom_menu = sg.Column([
                    [
                    sg.Column([[sg.Text('fontsize',background_color=color,text_color=text_color)],[sg.InputText(15,size=(7,1),key='fontsize')]],background_color=color),
                    sg.Column([[sg.Text('markersize',background_color=color,text_color=text_color)],[sg.InputText(5,size=(7,1),key='markersize')]],background_color=color),
                    sg.Column([[sg.Text('n_x_labels',background_color=color,text_color=text_color)],[sg.InputText(10,size=(7,1),key='n_x_labels')]],background_color=color),
                    sg.Column([[sg.Text('Sampling Transparency',background_color=color,text_color=text_color)],[sg.InputText(0.01,size=(7,1),key='sampling_alpha')]],background_color=color),
                    sg.Column([[sg.Text('Height (in)',background_color=color,text_color=text_color)],[sg.InputText(4.8,size=(7,1),key='height')]],background_color=color),
                    sg.Column([[sg.Text('Width (in)',background_color=color,text_color=text_color)],[sg.InputText(6.4,size=(7,1),key='width')]],background_color=color),

                    sg.Column([[sg.Text('Legend',background_color=color,text_color=text_color)],[sg.Checkbox('',key='legend',default=True,background_color=color,text_color=text_color)]],background_color=color),

                    sg.Column([[sg.Text('x_axis',background_color=color,text_color=text_color)],[sg.InputText('Wavelength (nm)',size=(30,1),key='xaxis')]],background_color=color),
                    sg.Column([[sg.Text('y_axis',background_color=color,text_color=text_color)],[sg.InputText('Normalized Intensity',size=(30,1),key='yaxis')]],background_color=color)

                ]
                    ],justification='center',background_color=color)

    layout = [
                [top_menu],
                [left_pannel,right_pannel],
                [bottom_menu]
            ]

    layout = [[sg.Column(layout,background_color=color,scrollable=True,expand_x=True,expand_y=True)]]

    #sg.ChangeLookAndFeel('default')
    window = sg.Window('Spectra Plotter',layout,finalize=True,background_color=color,resizable=True,icon=image_icon)#,element_justification='center')
    window['Mol1_name'].bind("<Return>", "")
    window['Mol2_name'].bind("<Return>", "")
    window['Mol3_name'].bind("<Return>", "")
    window['Mol4_name'].bind("<Return>", "")
    window['Mol5_name'].bind("<Return>", "")
    #right_pannel.expand()
    #left_pannel.expand()

    fig = plt.figure()#figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.set_xlim(200,600)
    ax.set_ylim(0,1)

    fig_agg = draw_figure(window['-CANVAS-'].TKCanvas,fig)

    inputs = [[]] * 5
    #print(inputs)
    equilibriums = [['']] * 5
    #print(equilibriums)
    names = [''] * 5
    #print(names)
    colors = [''] * 5
    #print(colors)


    while True:

        event,values = window.read()
        #print(event,values)
        if event in (None,'Exit'):
            break

        elif re.search('_inputs',event):
            index = int(event.split('_inputs')[0][-1]) - 1
            raw_inputs = values[event].split(';')
            checked_inputs = [input for input in raw_inputs if input != '']
            inputs[index] = checked_inputs

            #print('inputs updated')

        elif re.search('_equilibrium',event):
            index = int(event.split('_equilibrium')[0][-1]) - 1
            equilibriums[index] = values[event].split(';')

            #print('equilibriums updated')

        elif re.search('_name',event):
            index = int(event.split('_name')[0][-1]) - 1
            names[index] = values[event]

        elif event == 'plot':

            #clear out current figure so that we can resize
            ax.cla()
            fig_agg.get_tk_widget().forget()
            fig = plt.figure(figsize=(float(values['width']),float(values['height'])))
            ax = fig.add_subplot(111)
            ax.set_xlim(200,600)
            ax.set_ylim(0,1)

            fig_agg = draw_figure(window['-CANVAS-'].TKCanvas,fig)



            to_plot_inputs = []
            to_plot_equilibriums = []
            to_plot_names = []
            to_plot_colors = []

        #check for names that were updated without an event
            for index,manual_event in enumerate(['Mol1_name','Mol2_name','Mol3_name','Mol4_name','Mol5_name']):
                names[index] = values[manual_event]

            for index,manual_event in enumerate(['Mol1_color','Mol2_color','Mol3_color','Mol4_color','Mol5_color']):
                colors[index] = values[manual_event]

            loops = 0
            for index,(input,equilibrium,name,plot_color) in enumerate(zip(inputs,equilibriums,names,colors)):
                if len(input) > 0:
                    to_plot_inputs.append(input)
                    if name == '':
                        to_plot_names.append('Mol'+str(index+1))
                    else:
                        to_plot_names.append(name)
                    to_plot_equilibriums.append(equilibrium)
                    if plot_color == '':
                        if index > len(availible_colors) -1:
                            loops += 1
                            to_plot_colors.append(index - (loops * len(availible_colors)))
                        else:
                            to_plot_colors.append(index)
                    else:
                        to_plot_colors.append(plot_color)


            if len(to_plot_inputs) > 0:

                ax = plot_spectra(to_plot_inputs,to_plot_equilibriums,to_plot_names,ax,

                                start=int(values['start']),
                                end=int(values['stop']),
                                sigma=float(values['sigma']),
                                spacing=float(values['step']),
                                fontsize=int(values['fontsize']),
                                markersize=int(values['markersize']),
                                n_x_labels=int(values['n_x_labels']),
                                xaxis=values['xaxis'],
                                yaxis=values['yaxis'],
                                show_sampling=values['show_sampling'],
                                sampling_alpha=float(values['sampling_alpha']),
                                legend=values['legend'],
                                colors=to_plot_colors,
                                gui=True
                                )
                plt.tight_layout()
                fig_agg.draw()


        elif event == 'save':

            image_name = values['image_file']
            if image_name[-4:] != '.png':
                image_name += '.png'
            plt.savefig(image_name,bbox_inches='tight')

        elif event == 'help_info':

            help_col = sg.Column([[

            sg.Frame('HELP',[[

                sg.Text('''This program is designed to plot absorbance spectra from QM calculations.

                Right now only Gaussian output is accepted, but ORCA will also be added sometime in the future!
                _______________________________________________________________________________________________

                What you need:
                                - an output file from an excited state calculation, or a bunch of output files from excited state
                                calculations on multiple sampled geometries

                optional:
                                - an equilibrium structure to graph stems on the actual spectra


            Instructions:
                The left pannel is broken up into 5 blocks, each used to plot a separate molecule

                For as many molecules as you need to plot...

                    Choose the file, or files, that make up the spectra by browsing for them with the 'Excitations' button
                            - NOTE: if you are using sampled geometries, select ALL of the sampled geometries here!

                    This is all that is needed to generate a plot with the 'PLOT' button at the top right. However, you can additionally add:

                        - "Name": name of the molecule displayed on the legend
                        - "Equilibrium Excitation": displays just the stems for the excitations,
                            not a spectra, used to show difference between sampled and equilibrium spectra
                        - "color": color used to plot this molecule
                _______________________________________________________________________________________________

            You can adjust what exactly is being plotted with the top menu bar:
                    - "Show Sampling": if multiple files are selected by the "Excitations" button,
                        this will plot both the regular summed spectra as well as each individual spectra
                        with smaller and more transparent markers (see "Sampling Transparency" option on bottom toolbar)
                    - "sigma": the broadening parameter used
                    - "Start (nm)": the beginning of the x-axis range in nanometers
                    - "End (nm)": the end of the x-axis range in nanometers
                    - "Step (nm)": the step size for points calculated on the spectra
                    And the generated plot can be saved by providing a "Image FileName" and clicking the "SAVE" button


                Lastly, there are more fine-tuning parameters on the bottom toolbar for customizing the plot look and feel:
                    - "fontsize": axis label font size
                    - "markersize": size of markers
                    - "n_x_labels": number of labels on the x-axis
                    - "Sampling Transparency": how transparent are the sampled spectra when using "Show Sampling" option
                    - "Height (in)": height of the plot in inches
                    - "Width (in)": width of the plot in inches
                    - "Legend": choose to display or hide legend
                    - "x_axis": label for the x-axis
                    - "y_axis": label for the y-axis
    ''',background_color=color,text_color=text_color)]

            ],background_color=color,title_color=text_color)],


                [sg.Frame('INFO',[[sg.Text('''All of the calculations done behind the scenes are based on the equations here:
                    https://gaussian.com/uvvisplot/

                    Specifically, Equation 7 is used such that the spectra is computed as the sum of all of the excited state
                    gaussians across all files selected by the "Excitations" button
                _______________________________________________________________________________________________
                    Constants:

                    sigma defined by user

                    h=6.62607015e-34
                    cmeters=299792458
                    c=cmeters*100
                    eV2J=1.602176634e-19
                    sigmacm=((sigma*eV2J)/(h*c))  (converts sigma in eV to wavenumbers with 1/cm)
                _______________________________________________________________________________________________

                    Walking through the calculation:

                    wavelengths = np.arang(start,end,spacing) (array of wavelengths)

                    gauss = np.zeros((len(data),len(wavelengths)))  (empty 2d array with rows for each excited state and columns for each wavelength)

                    for i in range(0,len(wavelengths)):  (for every wavelength)
                        gauss[:,i]=1.3062974e8 * (data[:,1]/sigmacm) * np.exp(-(((1/wavelengths[i]-1/data[:,0])/(sigmamc*10e-8))**2))

                        (for every row, the column i is equal to 1.3062974e8 * (the oscialltor strength for that excitation / sigmacm)
                        * exponential[ - ((wavelength in wavenumbers ie 1/wavelength - wavelength of excitation in wavenumbers ie 1/excitaton wavelength) / sigmamc * 10e-8)**2
                        that last factor 10e-8 is to convert the wavelengths of nm to cm so the units will cancel with the 1/cm value of sigmacm


                    np.sum(gaus,axis=0) (then sum all excitations together to get 1 value per wavelength)


                _______________________________________________________________________________________________


                Normalization:

                    After ALL of the molecules on a plot have had their gauss arrays computed per the above section,
                    the maximum value across all molecules requested is used to normalize the spectra.

                    In other words, the largest peak on the graph will always hit 1 on the y-axis, and if multiple molecules are plotted, their absorbances as relative to each other.


                    If "Show Sampling" is selected, these are normalized separate from the overall summed plots (because the summed spectrum will have an absorbance much larger than the individual).
                    Similarly, the maximum of all the individuals across all molecules is computed so that everything is relative to one another!

                    NOTE: the equilibrium stems are not normalized since those are literally just the raw wavelength and oscillator strengths plotted


                    ''',background_color=color,text_color=text_color)



                ]],background_color=color,title_color=text_color)

            ]],scrollable=True,background_color=color,expand_x=True,expand_y=True)

            layout = [[help_col]]
            help_info_window = sg.Window('Help and Info',layout,finalize=True,background_color=color,resizable=True,icon=image_icon)#,element_justification='center')



    window.close()

if __name__ == '__main__':
    #get arguments
    parser = ArgumentParser()
    parser.add_argument('--gui',dest='gui',action='store_true',default=False)
    parser.add_argument('--start',dest='start',type=int,default=200)
    parser.add_argument('--end',dest='end',type=int,default=600)
    parser.add_argument('--sigma',dest='sigma',type=float,default=0.25)
    parser.add_argument('--spacing',dest='spacing',type=float,default=0.2)
    parser.add_argument('--title',dest='title',type=str)
    parser.add_argument('--fontsize',dest='fontsize',type=int,default=15)
    parser.add_argument('--markersize',dest='markersize',type=int,default=5)
    parser.add_argument('--n_x_labels',dest='n_x_labels',type=int,default=10)
    parser.add_argument('--nstates',dest='nstates',default=10,type=int)
    parser.add_argument('--xaxis',dest='xaxis',default='Wavelength (nm)',type=str)
    parser.add_argument('--no_legend',dest='legend',action='store_false',default=True)
    parser.add_argument('--sampling_alpha',dest='sampling_alpha',default=0.01,type=float)
    parser.add_argument('--yaxis',dest='yaxis',default='Normalized Intensity',type=str)
    parser.add_argument('--csv',dest='csv',default=False,action='store_true')
    parser.add_argument('--program',dest='program',default='infer',choices=['gaussian','orca','infer'])
    parser.add_argument('--show_sampling',dest='show_sampling',default=False,action='store_true')
    parser.add_argument('--image',dest='image_name',default=False,type=str)

    parser.add_argument('-i','--input',dest='inputs',action='append',nargs='+',type=str)
    parser.add_argument('-n','--name',dest='names',action='append',type=str,default=[])
    parser.add_argument('--equilibrium',dest='equilibriums',action='append',type=str,default=[])
    parser.add_argument('-c','--color',dest='colors',action='append',type=str,default=[],choices=availible_colors)

    arguments = parser.parse_args()

    if arguments.gui:
        Run_GUI()
    else:

        start = arguments.start
        end = arguments.end
        sigma = arguments.sigma
        spacing = arguments.spacing
        title = arguments.title
        fontsize = arguments.fontsize
        markersize = arguments.markersize
        n_x_labels = arguments.n_x_labels
        nstates = arguments.nstates
        xaxis = arguments.xaxis
        yaxis = arguments.yaxis
        csv = arguments.csv
        program = arguments.program
        show_sampling = arguments.show_sampling
        sampling_alpha = arguments.sampling_alpha
        legend = arguments.legend
        image_name = arguments.image_name

        inputs = arguments.inputs
        if inputs is None:
            raise IndexError('\nERROR! \n\nYou must provide an input!')        

        names = arguments.names
        equilibriums = [[equilibrium] for equilibrium in arguments.equilibriums]
        colors = arguments.colors

        if len(equilibriums) > 0:
            if len(equilibriums) != len(inputs):
                raise IndexError('\nERROR! \n\nIf --equilibrium option is used, there must be exactly 1 equilibrium file provided for each input')

        if len(names) > 0:
            if len(names) != len(inputs):
                raise IndexError('\nERROR! \n\nIf -n/--name option is used, there must be exactly 1 name provided for each input')

        if len(colors) > 0:
            if len(colors) != len(inputs):
                raise IndexError('\nERROR! \n\nIf -c/--color option is used, there must be exactly 1 color provided for each input')

        print('''
################
Spectra Plotter
################
* {0} - {1} nm with {2} step
* sigma: {3}
    '''.format(start,end,spacing,sigma))

        plot_spectra(inputs,equilibriums=equilibriums,names=names,
                                                start=start,
                                                end=end,
                                                sigma=sigma,
                                                spacing=spacing,
                                                title=title,
                                                fontsize=fontsize,
                                                markersize=markersize,
                                                n_x_labels=n_x_labels,
                                                nstates=nstates,
                                                xaxis=xaxis,
                                                yaxis=yaxis,
                                                csv=csv,
                                                program=program,
                                                show_sampling=show_sampling,
                                                sampling_alpha=sampling_alpha,
                                                legend=legend,
                                                image_name=image_name,
                                                colors=colors)

        print('Done!')

