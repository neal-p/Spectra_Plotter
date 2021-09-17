#! shared/centos7/anaconda3/2021.05/bin/python

from argparse import ArgumentParser
import PySimpleGUI as sg
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#get nice colors
tableau = [  (31, 119, 180), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (34, 189, 184), (141, 178,220), (23, 190, 207), (158, 218, 229),
             (31, 119, 180), (233, 175, 175), (255, 127, 14), (255, 215, 121),(0,0,0)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau)):
    r, g, b = tableau[i]
    tableau[i] = (r / 255., g / 255., b / 255.)

color_to_index = {'':None,
                  'blue':0,
                  'orange':1,
                  'light_orange':2,
                  'green':3,
                  'light_green':4,
                  'red':5,
                  'pink':6,
                  'purple':7,
                  'light_purple':8,
                  'brown':9,
                  'tan':10,
                  'hot_pink':11,
                  'light_pink':12,
                  'grey':13,
                  'light_grey':14,
                  'aqua':15,
                  'blue_grey':16,
                  'teal':17,
                  'light_blue':18,
                  'dark_blue':19,
                  'salmon':20,
                  'bright_orange':21,
                  'gold':22,
                  'black':23

                        }

availible_colors = list(color_to_index.keys())

#######################################################################################################################

def GUI_error(message):
   title = 'ERROR!!!'
   sg.Popup(message,title=title)

def read_gaussian(files,df_name,nstates,csv=False,gui=False):
    '''Read the excited state info for a list of files'''

    individuals = []
    all = pd.DataFrame()

    for file in files:
        found = False
        with open(file,'r') as log:
            output_lines = log.read().splitlines()
            for line_number,line in enumerate(output_lines):
                if re.search('Excitation energies and oscillator strengths',line):
                    found = True

                    states = []

                    line_number += 2 #get to the first Excited State line
                    line = output_lines[line_number]

                    while re.search('Excited State',line):
                        state_dict = {}
                        line = line.split()

                        state_dict['character'] = line[3]
                        state_dict['energy'] = float(line[4])
                        state_dict['wavelength'] = float(line[6])
                        state_dict['f'] = float(line[8].split('=')[-1])

                        #get all of the orbital transitions
                        line_number += 1
                        line = output_lines[line_number]
                        transitions = []
                        while re.search('->',line):
                            line = line.split()
                            transitions.append(('->'.join([line[0],line[2]]),float(line[3])))
                            line_number += 1
                            line = output_lines[line_number]

                        state_dict['transitions'] = transitions.copy()
                        states.append(state_dict.copy())

                        #find the next excited state line
                        #there is either a blank line,
                        #or a message about the state for optimization,
                        #then a blank line in between

                        if re.search('This state for optimization',line):
                            line_number += 4
                            line = output_lines[line_number]

                        else:
                            line_number += 1
                            line = output_lines[line_number]
                   #keep only the number of states requested
                    states = states[0:nstates]
                    df = pd.DataFrame(states)
                    df['origin'] = file
            if not found:
                if gui:
                    GUI_error(f'No excited state info found in {file}')
                    return(None)
                raise IndexError(f'No excited state info found in {file}')
 
            print('    Read Excited States from: {0}'.format(file),flush=True)

        individuals.append(df)
        all = pd.concat([all,df])

    if csv:
        all.to_csv(df_name+'.csv',index=False)

    individuals = [individual[['wavelength','f']].to_numpy() for individual in individuals]

    #return the combined and the individual for plotting individual sampled structures
    return(individuals)

def read_orca(files,df_name,nstates,csv=False,gui=False):
    '''Read the excited state info for a list of files'''
    if gui:
        GUI_error('parsing ORCA is not finished yet! Please only use Gaussian inputs for now...')
        return(None)
    raise NotImplementedError('parsing ORCA is not finished yet!')

def calc_linear_comb(data,wavelengths,sigmacm):
    '''converts the raw excitation parsed from output files to values to plot at each wavelength'''

    gauss=np.zeros((len(data),len(wavelengths)))

    for i in range(0,len(wavelengths)):
        gauss[:,i]=1.3062974e8 * (data[:,1]/sigmacm) * np.exp(-(((1/wavelengths[i]-1/data[:,0])/(sigmacm*10e-8))**2))

    return(np.sum(gauss,axis=0))



def plot_spectra(inputs,equilibriums=[],names=[],ax1=None,start=200,
                                              end=600,
                                              sigma=0.25,
                                              title=None,
                                              spacing=0.2,
                                              fontsize=15,
                                              markersize=5,
                                              n_x_labels=10,
                                              nstates=10,
                                              xaxis='Wavelength (nm)',
                                              yaxis='Normalized Intensity',
                                              csv=False,
                                              program='infer',
                                              show_sampling=False,
                                              sampling_alpha=0.01,
                                              legend=True,
                                              colors=[],
                                              gui=False,
                                              image_name=False):
    '''Read files and plot spectra'''

    #constants and setup
    h=6.62607015e-34
    cmeters=299792458
    c=cmeters*100
    eV2J=1.602176634e-19
    sigmacm=((sigma*eV2J)/(h*c))

    if ax1 is None:
        fig =  plt.figure()
        ax1 = fig.add_subplot(111)
    else:
        ax1.cla()

    wavelengths = np.arange(start,end,spacing)
    ev=1240/wavelengths


    #if no names are provided, make some names for internal tracking, but don't actually visualize them
    if len(names) < 1:
        names = [f'mol{index}' for index,input in enumerate(inputs)]

    for name,input in zip(names,inputs):
        print(f'    Input: {name} ... {len(input)} files to read')

    #fill in colors if not passed
    if len(colors) < 1:
        loops = 0
        for index,input in enumerate(inputs):
            if index > len(tableau)-1:
                loops += 1
                colors.append(tableau[index - (loops * len(tableau))])
            else:
                colors.append(tableau[index])

    #if colors are passed, convert them all to rgbs or hex
    else:
        colors_parsed = []
        for color in colors:
            try:
                colors_parsed.append(tableau[int(color)])
            except:
                if re.search(',',color):
                    rgbs = color.split(',')
                    try:
                        rgbs = [int(rgb.strip()) for rgb in rgbs]
                    except:
                        if gui:
                            GUI_error(f'"{rgbs}" is not a valid rgb triplet. It should be of the format "0,0,0"')
                            ax1.cla()
                            return(ax1)
                        raise TypeError(f'"{rgbs}" is not a valid rgb triplet. It should be of the format "0,0,0"')
                    rgbs = rgbs[0:3]
                    maximum = float(max(rgbs))
                    rgbs = tuple(rgbs)
                    r,g,b = rgbs
                    if maximum > 1:
                        colors_parsed.append((r / maximum, g / maximum, b / maximum))
                    else:
                        colors_parsed.append((r,g,b))
                elif re.search('#',color):
                    if not re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$',color):
                        if gui:
                            GUI_error(f'"{color}" is not a valid hex color')
                            ax1.cla()
                            return(ax1)
                        raise TypeError(f'"{color}" is not a valid hex color')
                    colors_parsed.append(color)
                else:
                    if color in color_to_index:
                        colors_parsed.append(tableau[color_to_index[color]])
                    else:
                        if gui:
                            GUI_error(f'"{color}" is not a valid rgb triplet, not a valid hex color, and is not in the list of availible colors!')
                            ax1.cla()
                            return(ax1)
                        raise TypeError(f'{color} is not a valid rgb triplet, not a valid hex color, and is not in the list of availible colors!')
        colors = colors_parsed[:]

    #set bool for having to do equilibriums
    if len(equilibriums) > 0:
        do_equilibriums = True
    else: 
        do_equilibriums = False

    #get the impulses
    #first need to decide what program's output we are reading
    if program == 'infer':
        #if the extension is .out -> ORCA
        #if the extension is .log -> gaussian
        if inputs[0][0].split('.')[-1] == 'out':
            program = 'orca'
        elif inputs[0][0].split('.')[-1] == 'log':
            program = 'gaussian'
        else:
            if gui:
                GUI_error('\nERROR!! \n\nCould not figure out if using gaussian or orca!\n\nPlease rename to .log extension')
                ax1.cla()
                return(ax1)

            raise ValueError('\nERROR!! \n\nCould not figure out if using gaussian or orca!\n\nPlease specify manually with the --program option')

    #now parse with the correct parser 
    #we are handed a list of lists,
        #the outer list is a list of molecules
            #the inner list is a list of files for each molecule

        #we need to extract out both a datastructure of all of impulses in all files, as well as impulses in each individual
        #then we only plot the individual if the show_sampling is True
    

    #maybe for now, just get everything as individuals, then can concatenate into all when ready

    if program == 'gaussian':
        print('Parsing log files...')
        individual_impulses = [read_gaussian(input,name,nstates,csv,gui) for input,name in zip(inputs,names)]
        if None in individual_impulses:
            ax1.cla()
            return(ax1)
        if do_equilibriums: 
            #take the first bc read_gaussian returns a list, but only expect 1 element
            equilibrium_impulses = [read_gaussian(equilibrium,name+'_eq',nstates,csv,gui) if equilibrium != [''] else 'skip' for equilibrium,name in zip(equilibriums,names)]
            if None in equilibrium_impulses:
                ax1.cla()
                return(ax1)
            equilibrium_impulses = [equilibrium[0] if equilibrium != 'skip' else equilibrium for equilibrium in equilibrium_impulses]


    elif program == 'orca':
        print('Parsing out files...')
        individual_impulses = [read_orca(input,name,nstates,csv,gui) for input,name in zip(inputs,names)]
        if None in equilibrium_impulses:
            ax1.cla()
            return(ax1)
        if do_equilibriums:
            #take the first bc read_orca returns a list, but only expect 1 element
            equilibrium_impulses = [read_orca(equilibrium,name+'_eq',nstates,csv,gui) if equilibrium != [''] else 'skip' for equilibrium,name in zip(equilibriums,names)]
            if None in equilibrium_impulses:
                ax1.cla()
                return(ax1)
            equilibrium_impulses = [equilibrium[0] if equilibrium != 'skip' else equilibrium for equilibrium in equilibrium_impulses]

    else:
        if gui:
            GUI_error('\nERROR!! \n\nCould not figure out if using gaussian or orca!\n\nPlease rename to .log extension')
            ax1.cla()
            return(ax1)

        raise ValueError('\nERROR!! \n\nCould not figure out if using gaussian or orca!\n\nPlease specify manually with the --program option')

    print('\nComputing spectra...')

    #get the values for plotting based on the summed spectra,
    #use this as the normalization
 
    #1d list of 1 array per molecule
    linear_combs_summed = [calc_linear_comb(np.concatenate(impulse,axis=0),wavelengths,sigmacm) for impulse in individual_impulses]
    #2d list of list of arrays per molecule
    linear_combs_individual = [[calc_linear_comb(individual,wavelengths,sigmacm) for individual in impulses] for impulses in individual_impulses]

    #normalize to the maximum for normalization based on the summed arrays
    maximum = 0
    for linear_comb in linear_combs_summed:
        local_maximum = np.amax(linear_comb)
        if local_maximum > maximum:
            maximum = local_maximum

    #normalize the sampling maxima too
    if show_sampling:
        sampling_maximum = 0
        for linear_comb in linear_combs_individual:
            for individual in linear_comb:
                local_maximum = np.amax(individual)
                if local_maximum > sampling_maximum:
                    sampling_maximum = local_maximum


    #Start actually plotting things

    #plot individual sampling peaks
    if show_sampling:
        loops = 0
        for outer_index,(impulses,name,color) in enumerate(zip(linear_combs_individual,names,colors)):
            print('    Plotting: {0} individual sampled spectra'.format(name))
            for inner_index,individual in enumerate(impulses):
                #be sure to normalize by the maximum calculated earlier
                ax1.scatter(wavelengths,individual/sampling_maximum,s=markersize/2, color=color, alpha=sampling_alpha)

    #plotting equilibrium stems
    if do_equilibriums:
        loops = 0
        for index,(data,name,color) in enumerate(zip(equilibrium_impulses,names,colors)):
            if data != 'skip':
                print('    Plotting: {0} equilibrium stems'.format(name))
                markerline, stemlines, baseline=ax1.stem(data[:,0],data[:,1], linefmt=None, markerfmt=',', basefmt=None)
                plt.setp(stemlines, 'color', color)

    #now plot the normal summed spectra last
    loops = 0
    for index,(linear_comb,name,color) in enumerate(zip(linear_combs_summed,names,colors)):
        print('    Plotting: {0} spectrum'.format(name))
        #be sure to normalize by the maximum calculated earlier
        ax1.scatter(wavelengths,linear_comb/maximum,s=markersize, color=color, label=name, alpha=0.5)

    plt.locator_params(axis='x', nbins=n_x_labels)
    if legend:
        plt.legend()
    ax1.set_xlabel(xaxis,fontsize=fontsize)
    ax1.set_ylabel(yaxis,fontsize=fontsize)
    ax1.set_xlim(start,end)
    plt.title(title)

    if gui:
        return(ax1)

    if not image_name:
        image_name = 'my_spectra'        

    if image_name[-4:] != '.png':
        image_name += '.png'
        
    plt.savefig(image_name,bbox_inches='tight')
    print(f'\nSaved plot to {image_name}')


if __name__ == '__main__':

    #get arguments
    parser = ArgumentParser()
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
                                               image_name=image_name)

    print('Done!')


