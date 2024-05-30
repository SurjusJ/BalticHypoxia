# # PYTHON
# !pip install numpy
# !pip install pandas
# !pip install matplotlib

##### PACKAGES #####
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from copy import deepcopy
import matplotlib.colors as mcolors


###############################################################################
############################## CALLING THE FILES ##############################
###############################################################################
##### LOCATION OF ALL THE DATA #####
# LOCATION_Data=r"..\Python" 
LOCATION_Data=r"D:\JJJJJJJ\Documents\STAGE-EAWAG\BalticHypoxia-main\Python"    # Path for oxygen 

##### OPEN FILES ######
# --> Oxygen
OXY = 'OXY.csv'                                                                # File name
chemin_complet = os.path.join(LOCATION_Data, OXY)                              # Complete path
OXY=pd.read_csv(chemin_complet)                                                # Openning file 

OXY_GLOBAL = 'OXY_GLOBAL.csv'                                                 
chemin_complet = os.path.join(LOCATION_Data, OXY_GLOBAL)                       
OXY_GLOBAL=pd.read_csv(chemin_complet)

# --> Flounder (Skrubba)
SK_OXY = 'SK_OXY.csv'                                                          
chemin_complet = os.path.join(LOCATION_Data, SK_OXY)                        
SK_OXY=pd.read_csv(chemin_complet)  

SK_OXY_GLOBAL = 'SK_OXY_GLOBAL.csv'                                      
chemin_complet = os.path.join(LOCATION_Data, SK_OXY_GLOBAL)               
SK_OXY_GLOBAL=pd.read_csv(chemin_complet) 

# --> Cod (Torsk)
TO_OXY = 'TO_OXY.csv'                                                        
chemin_complet = os.path.join(LOCATION_Data, TO_OXY)                    
TO_OXY=pd.read_csv(chemin_complet)  

TO_OXY_GLOBAL = 'TO_OXY_GLOBAL.csv'                                          
chemin_complet = os.path.join(LOCATION_Data, TO_OXY_GLOBAL)               
TO_OXY_GLOBAL=pd.read_csv(chemin_complet)

# --> Trawl Torsk
TRAWL_TO_OXY = 'TRAWL_TO_OXY.csv'                                                
chemin_complet = os.path.join(LOCATION_Data, TRAWL_TO_OXY)               
TRAWL_TO_OXY=pd.read_csv(chemin_complet)  

TRAWL_TO_OXY_GLOBAL = 'TRAWL_TO_OXY_GLOBAL.csv'                         
chemin_complet = os.path.join(LOCATION_Data, TRAWL_TO_OXY_GLOBAL)       
TRAWL_TO_OXY_GLOBAL=pd.read_csv(chemin_complet)

# --> Stomach Cod
STO_COD_OXY = 'STO_COD_OXY.csv' 
chemin_complet = os.path.join(LOCATION_Data, STO_COD_OXY)     
STO_COD_OXY=pd.read_csv(chemin_complet)  

STO_COD_OXY_GLOBAL = 'STO_COD_OXY_GLOBAL.csv'
chemin_complet = os.path.join(LOCATION_Data, STO_COD_OXY_GLOBAL)
STO_COD_OXY_GLOBAL=pd.read_csv(chemin_complet)

# --> Stomach Fle
STO_FLE_OXY = 'STO_FLE_OXY.csv'    
chemin_complet = os.path.join(LOCATION_Data, STO_FLE_OXY)
STO_FLE_OXY=pd.read_csv(chemin_complet)  

STO_FLE_OXY_GLOBAL = 'STO_FLE_OXY_GLOBAL.csv'    
chemin_complet = os.path.join(LOCATION_Data, STO_FLE_OXY_GLOBAL)
STO_FLE_OXY_GLOBAL=pd.read_csv(chemin_complet)

# --> Abondance Sprat
AB_S_OXY= 'AB_S_OXY.csv'                 
chemin_complet = os.path.join(LOCATION_Data, AB_S_OXY)
AB_S_OXY=pd.read_csv(chemin_complet)  

AB_S_OXY_GLOBAL = 'AB_S_OXY_GLOBAL.csv'
chemin_complet = os.path.join(LOCATION_Data, AB_S_OXY_GLOBAL)
AB_S_OXY_GLOBAL=pd.read_csv(chemin_complet)

# --> Abondance Herring
AB_H_OXY= 'AB_H_OXY.csv' 
chemin_complet = os.path.join(LOCATION_Data, AB_H_OXY) 
AB_H_OXY=pd.read_csv(chemin_complet)  

AB_H_OXY_GLOBAL = 'AB_H_OXY_GLOBAL.csv' 
chemin_complet = os.path.join(LOCATION_Data, AB_H_OXY_GLOBAL)   
AB_H_OXY_GLOBAL=pd.read_csv(chemin_complet)



############################## SET UP THE INDEX ##############################

" This function sets up the index of the FILE_OXY . "

def INDEX(FILE):
    colnames=list(FILE.columns)
    colnames[0]="Year - ICES"
    FILE.columns=colnames
    FILE=FILE.set_index("Year - ICES")
    return FILE


" This function sets up the index of the FILE_OXY_GLOBAL . "

def INDEX_GLOBAL(FILE):
    colnames=list(FILE.columns)
    colnames[0]="Year"
    FILE.columns=colnames
    FILE=FILE.set_index("Year")
    return FILE

# --> Oxygen
OXY = INDEX(OXY)
OXY_GLOBAL = INDEX_GLOBAL(OXY_GLOBAL)
# --> Skrubba
SK_OXY = INDEX(SK_OXY)                                                   
SK_OXY_GLOBAL = INDEX_GLOBAL(SK_OXY_GLOBAL)    
# --> Torsk                                            
TO_OXY = INDEX(TO_OXY)                                                     
TO_OXY_GLOBAL = INDEX_GLOBAL(TO_OXY_GLOBAL)     
# --> Trawl Torsk                                       
TRAWL_TO_OXY = INDEX(TRAWL_TO_OXY)                                                  
TRAWL_TO_OXY_GLOBAL = INDEX_GLOBAL(TRAWL_TO_OXY_GLOBAL) 
# --> Stomach Cod                              
STO_COD_OXY = INDEX(STO_COD_OXY)                                                     
STO_COD_OXY_GLOBAL = INDEX_GLOBAL(STO_COD_OXY_GLOBAL)    
# --> Stomach Flounder                                            
STO_FLE_OXY = INDEX(STO_FLE_OXY)                                       
STO_FLE_OXY_GLOBAL = INDEX_GLOBAL(STO_FLE_OXY_GLOBAL)        
# --> Abundance Sprat                                            
AB_S_OXY = INDEX(AB_S_OXY)                                       
AB_S_OXY_GLOBAL = INDEX_GLOBAL(AB_S_OXY_GLOBAL)    
# --> Abundance Herring
AB_H_OXY = INDEX(AB_H_OXY)                                       
AB_H_OXY_GLOBAL = INDEX_GLOBAL(AB_H_OXY_GLOBAL)     



################################ OXYGEN COLORS ################################
# --> Association colors and parameters
colnames=list(OXY.columns)
colors=["lightblue","darkblue","red","darkred"]
colors=pd.DataFrame([colors],columns=colnames) 










###############################################################################
###############################################################################
################################### OXYGEN ###################################
###############################################################################
###############################################################################

"This Function plot the evolution of area and volume of hypoxic zone by year"

def PLOT_Evolution_OXY_by_Average(oxy_PLOT_mean_global):        
    ##### LABELS #####
    labels={"A1ml":"[O2]<1ml",
            "V1ml":"[O2]<1ml",
            "A43ml":"[O2]<43ml",
            "V43ml":"[O2]<43ml"}
    
    ##### FIGURES #####
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 9))
    fig.suptitle("OXYGENE")
    
    for k in [0,1]:
        if k==0:
            # --> Plotting of the areas
            col="blue"
            axes[k].plot(oxy_PLOT_mean_global.iloc[:,0], marker=' ', color=colors.iloc[0, 0], label=labels[str(oxy_PLOT_mean_global.columns[0])])
            axes[k].plot(oxy_PLOT_mean_global.iloc[:,1], marker=' ', color=colors.iloc[0, 1], label=labels[str(oxy_PLOT_mean_global.columns[1])])
            axes[k].set_title('Evolution of the average area of hypoxic seafloor by years')
            axes[k].set_ylabel('Areas (km2)')
            
        else :
            # --> Plotting of the Volumes
            col="red"
            axes[k].plot(oxy_PLOT_mean_global.iloc[:,2], marker=' ', color=colors.iloc[0, 2], label=labels[str(oxy_PLOT_mean_global.columns[2])])
            axes[k].plot(oxy_PLOT_mean_global.iloc[:,3], marker=' ', color=colors.iloc[0, 3], label=labels[str(oxy_PLOT_mean_global.columns[3])])
            axes[k].set_title('Evolution of the average volume of hypoxic seafloor by years')
            axes[k].set_ylabel('Volumes (km3)')
            
        # --> Abscissa
        axes[k].set_xlabel('Years')
        axes[k].legend(loc="lower right", prop={'size': 15})
        
        # --> Customizing tick labels font size
        axes[k].tick_params(axis='x', labelsize=15, rotation=45)
        axes[k].tick_params(axis='y', labelsize=15)
        
        # --> Size and thickness of abscissa axes
        axes[k].xaxis.label.set_fontsize(15) 
        axes[k].xaxis.label.set_fontweight('bold')  
        
        # --> Color, size and thickness of ordinate axes
        axes[k].yaxis.label.set_color(col)
        axes[k].yaxis.label.set_fontsize(15) 
        axes[k].yaxis.label.set_fontweight('bold')  

    ##### LEGEND #####    
    axes[0].text(0.02, 0.98, 'A', transform=axes[0].transAxes, fontsize=16, fontweight='bold', va='top')
    axes[1].text(0.02, 0.98, 'B', transform=axes[1].transAxes, fontsize=16, fontweight='bold', va='top')
    
    ##### RESULT ##### 
    return()



################################### FIGURE ###################################
# --> Shows the evolution of the average of each parameters by years on oxygene data
PLOT_Evolution_OXY_by_Average(OXY_GLOBAL)










###############################################################################
###############################################################################
################################### WEIGHT ###################################
###############################################################################
###############################################################################


###############################################################################
############################ OXYGEN DATA SELECTION ############################
###############################################################################

" This function selects oxygen datas in the same sampling area than the species. "
" Also, selects oxygene datas and calculates the average for each year. "

def OXY_by_ICES(FILE_OXY):

    ################################## FILE ##################################
    # --> Correct colnames
    FILE_OXY=deepcopy(FILE_OXY)
    #FILE_OXY=Colnames(FILE_OXY)
    
    # --> Set up the index
    FILE_OXY[['Year', 'ICES']] = FILE_OXY["Year - ICES"].str.split(' - ',expand=True)
    del(FILE_OXY["Year - ICES"])
    FILE_OXY=FILE_OXY.set_index("ICES")
    # --> Stores YEARS list
    YEARS=sorted(list(set(FILE_OXY["Year"])))
    
    
    ################################# OXYGEN #################################
    ##### New file for oxygen datas and index set-up #####
    OXY=FILE_OXY.iloc[:,0:4]
    OXY["Year"]=FILE_OXY["Year"]
    OXY=OXY.set_index("Year")
    
    ##### New file for oxygen : stores average values for each year #####
    SPECIES_OXY=np.zeros([len(YEARS),OXY.shape[1]])  
    # --> Set up the index : all the different years
    index = pd.Series(YEARS)
    SPECIES_OXY=pd.DataFrame(SPECIES_OXY,index=index)
    # --> Adding columns names
    SPECIES_OXY.columns=list(OXY.columns)
    
    ##### Average for each year #####
    for year in YEARS:
        interest=OXY.loc[year]                                                      # ... Selection of all of the data for the same year
        if type(interest)==pd.core.series.Series:                                   # ... If there is only one row for this year ...
            SPECIES_OXY.loc[year]=np.array([interest])                                  # ... Stores this row in the new file
        else :                                                                      # ... Else ...
            SPECIES_OXY.loc[year]=np.array([interest.mean(axis=0)])                     # Stores the average of each parameters for this year
    
    
    ################################# RETURN #################################  
    return (SPECIES_OXY)





###############################################################################
####################### WEIGHT DATA SELECTION & AVERAGE #######################
###############################################################################

" This function selects weight predators and preys datas. "
" Also, selects weight datas for each year-ICES and calculates the average for each combinaison. "

def Weigth_by_ICES(FILE_OXY):
    
    FILE_OXY=deepcopy(FILE_OXY)
    
    # --> Set up the index
    FILE_OXY[['Year', 'ICES']] = FILE_OXY["Year - ICES"].str.split(' - ',expand=True)
    del(FILE_OXY["Year - ICES"])
    # --> Stores ICES list
    ICES=sorted(list(set(FILE_OXY["ICES"])))
    FILE_OXY=FILE_OXY.set_index("ICES")
    # --> Stores YEARS list
    YEARS=sorted(list(set(FILE_OXY["Year"])))
    

    ################################# WEIGHT #################################
    ##### WEIGHT DATAS #####
    FILE_OXY = FILE_OXY.loc[:, ['Year', 'Weight', 'Prey weight']]
    
    ##### New file for WEIGHT #####
    # --> Stores the number of 0 for each year by ICES
    WEIGHT_PRED=pd.DataFrame(index=YEARS)
    # --> Stores the number of 0 for each year by ICES
    WEIGHT_PREY=pd.DataFrame(index=YEARS)
    
    ##### Average for each year by ICES #####
    for ices in ICES:
        interest=FILE_OXY.loc[ices]                                                  # ... Selection of all of the data for the same ices
        
        # --> Conversion of Series data in DataFrame
        if type(interest)==pd.core.series.Series: 
            interest=FILE_OXY.loc[ices].to_frame()                                     
            interest=interest.T
          
        # --> Set up the index
        interest=interest.set_index("Year")   
              
        ##### VALUES #####
        # --> Stores the average of proportion prey for each year on this ICES
        WEIGHT_PRED[str(ices)]=interest["Weight"]
        WEIGHT_PREY[str(ices)]=interest["Prey weight"]
        

    ################################# RETURN #################################    
    return (WEIGHT_PRED,WEIGHT_PREY)





###############################################################################
########################## MEAN AND MEDIAN OF WEIGHT ##########################
###############################################################################

" This function calculates the mean and the median for the parameter for each year. "

def SPECIES_n_ZEROS_by_YEARS(FILE_ZEROS , FILE_VALUES):  
    
    colors["Mean"]="lightgreen"
    colors["Median"]="darkgreen"
    
    FILE_ZEROS_GLOBAL=pd.DataFrame()
    # --> MEAN
    FILE_ZEROS_GLOBAL["Mean"] = FILE_ZEROS.mean(axis=1)
    # --> MEDIAN
    FILE_ZEROS_GLOBAL["Median"] = FILE_ZEROS.apply(lambda row: row.median(), axis=1)
    
    FILE_VALUES_GLOBAL=pd.DataFrame()
    # --> MEAN
    FILE_VALUES_GLOBAL["Mean"] = FILE_VALUES.mean(axis=1)
    # --> MEDIAN
    FILE_VALUES_GLOBAL["Median"] = FILE_VALUES.apply(lambda row: row.median(), axis=1)
    
    return(FILE_ZEROS_GLOBAL , FILE_VALUES_GLOBAL)





###############################################################################
################################ WEIGHT GRAPHS ################################
###############################################################################

" This function plots oxygen (area datas) and species datas. "

def PLOT_PARAM(FILE_OXY , FILE_ZEROS , FILE_VALUES , file_title , Graph_title , Graph_axis_title):   
  
    ################################# COLORS ################################# 
    rang = FILE_VALUES.shape[1]
    green = ["lightgreen","darkgreen"]
    green += ['#%02x%02x%02x' % (0, rd.randint(100, 255), 0) for _ in range(rang-2)]
    
    ################################# FIGURE #################################
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 9))
    fig.suptitle("AVERAGE FOR OXYGENE AND "+Graph_title)
    fig.text(0.5, 0.92, file_title, ha='center',fontweight='bold')
    
    ##### SPECIES #####
    # --> 2nd axis: Parameter axis Legend (on the right) for the species
    axes2_0 = axes[0].twinx()
    axes2_1 = axes[1].twinx()  
    
    # --> Stores legend color
    handles = []
    # --> Stores legend name
    labels = []   
    
    for ices in range(FILE_VALUES.shape[1]):
        ################################ ZEROS ################################ 
        line_param_0, = axes2_0.plot(FILE_ZEROS.iloc[:,ices], marker=' ', color=green[ices])
        
        ############################# PROPORTIONS ############################# 
        line_param_1, = axes2_1.plot(FILE_VALUES.iloc[:,ices], marker=' ', color=green[ices])
        
        ############################### LEGEND ###############################
        # --> Store new legend elements and their color
        handles.append(plt.Line2D([0], [0], color=green[ices]))
        labels.append(FILE_ZEROS.columns[ices])
    
    # --> Legend for the second axis of each graph
    axes2_0.set_ylabel(Graph_axis_title[0])           
    axes2_0.tick_params(axis='y', labelsize=15)
    # --> Color, size and thickness of ordinate axes
    axes2_0.yaxis.label.set_color("green")
    axes2_0.yaxis.label.set_fontsize(15) 
    axes2_0.yaxis.label.set_fontweight('bold')  

    axes2_1.set_ylabel(Graph_axis_title[1])              
    axes2_1.tick_params(axis='y', labelsize=15)
    # --> Color, size and thickness of ordinate axes
    axes2_1.yaxis.label.set_color("green")
    axes2_1.yaxis.label.set_fontsize(15) 
    axes2_1.yaxis.label.set_fontweight('bold')  

    
    ##### OXYGEN #####
    # --> Labels
    Labels={"A1ml":"[O2]<1ml",
            "V1ml":"[O2]<1ml",
            "A43ml":"[O2]<43ml",
            "V43ml":"[O2]<43ml"}
    
    # --> For each side
    for k in [0,1]:
        ################################ AREAS ################################ 
        label=['Area of hypoxic zone (km2)',"A43ml", "A1ml"]
        col="blue"
        #title='Evolution of the average area of hypoxic seafloor by years'
        line1, = axes[k].plot(FILE_OXY.iloc[:,0], marker=' ', color=colors.iloc[0, 0])
        line2, = axes[k].plot(FILE_OXY.iloc[:,1], marker=' ', color=colors.iloc[0, 1])
            
        # --> Store new legend elements and their color
        if k==0:
            handles.extend([plt.Line2D([0], [0], color=colors.iloc[0, 0]),
                                plt.Line2D([0], [0], color=colors.iloc[0, 1])])
            labels.extend([Labels['A43ml'], Labels['A1ml']])

        # --> Legend for the first axis of both graphs
        axes[k].set_title("Evolution of "+Graph_axis_title[k]+" by years")
        axes[k].set_xlabel('Years')
        axes[k].set_ylabel(label[0])
        
        # --> Size and thickness of abscissa axes
        axes[k].xaxis.label.set_fontsize(15) 
        axes[k].xaxis.label.set_fontweight('bold')  
        
        # --> Color, size and thickness of ordinate axes
        axes[k].yaxis.label.set_color(col)
        axes[k].yaxis.label.set_fontsize(15) 
        axes[k].yaxis.label.set_fontweight('bold')  
        
        # Customizing tick labels font size
        axes[k].tick_params(axis='x', labelsize=15, rotation=45)
        axes[k].tick_params(axis='y', labelsize=15)
        
        
    ##### LEGEND #####
    fig.legend(handles, labels, loc='lower center', prop={'size': 15}, bbox_to_anchor=(0.5, -0.009)) #, 'weight': 'bold'
    axes[0].text(0.02, 0.98, 'A', transform=axes[0].transAxes, fontsize=16, fontweight='bold', va='top')
    axes[1].text(0.02, 0.98, 'B', transform=axes[1].transAxes, fontsize=16, fontweight='bold', va='top')
    
    ##### FIGURE #####
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)

    ################################# RETURN #################################
    return ()





###############################################################################
####################### WEIGHT CALCULATION AND PLOTTING #######################
###############################################################################

" This function extract Mean and Median weight datas for each years. "

def WEIGHT(FILE_OXY , STO_FILE_OXY , Graph_subtitle):
    ##### SPECIE #####
    # --> Each ICES
    FILE_WEIGHT_PRED , FILE_WEIGHT_PREY = Weigth_by_ICES(STO_FILE_OXY)
    # --> Average ICES
    FILE_WEIGHT_PRED_GLOBAL , FILE_WEIGHT_PREY_GLOBAL = SPECIES_n_ZEROS_by_YEARS(FILE_WEIGHT_PRED , FILE_WEIGHT_PREY)
   
    ################################### WEIGHT ###################################
    PLOT_PARAM(FILE_OXY , FILE_WEIGHT_PRED_GLOBAL , FILE_WEIGHT_PREY_GLOBAL , Graph_subtitle , "WEIGHT" , ["Predator Weight","Prey Weight"])
  
    return()



###############################################################################
################################### FIGURES ###################################
###############################################################################

################################## TREATMENT ################################## 
##### COLNAMES #####
STO_COD_OXY["Year - ICES"]=STO_COD_OXY.index
STO_FLE_OXY["Year - ICES"]=STO_FLE_OXY.index


###### OXYGEN ######
# --> Selection of Oxygen datas
### COD ###
COD_OXY = OXY_by_ICES(STO_COD_OXY)

### FLE ###
FLE_OXY = OXY_by_ICES(STO_FLE_OXY)


################################### FIGURES ###################################
##### COD & AREA #####
WEIGHT(COD_OXY , STO_COD_OXY , "COD")

##### FLOUNDER & AREA #####
WEIGHT(FLE_OXY , STO_FLE_OXY , "FLOUNDER")










###############################################################################
###############################################################################
################################## ABUNDANCE ##################################
###############################################################################
###############################################################################


###############################################################################
################################ ONE PARAMETER ################################
###############################################################################

"This Function plot the evolution of area and volume of hypoxic zone and of one other paremeter by year"

def PLOT_Evolution_Data_by_Average(FILE_OXY_GLOBAL,parameters,file_title): 
    
    ############################## PARAMETERS ################################
    # --> Selection of oxygen data
    oxy=FILE_OXY_GLOBAL.iloc[:,0:4]
    
    # --> Storing the number of different parameters
    rang=parameters[-1]-parameters[0]
    
    # --> Selection of data for the parameters choosed
    parameters=FILE_OXY_GLOBAL.iloc[:,parameters[0]:parameters[-1]]
    # --> Stores the parameter name
    if isinstance(parameters, pd.core.series.Series):
        print("R")
    else:
        axe_title=parameters.columns[0]
        axe_title=axe_title.split(':')[0].strip()
    
    # --> Several shades of green 
    cmap = plt.cm.get_cmap('Greens')                                           # Green colormap creation
    vmin=30                                                                    # Maximal  intensity
    vmax=75                                                                    # Minimal  intensity
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)                             # Green gradient creation (size=rang)
    gradient = np.linspace(vmax, vmin, rang+2)
    green = [mcolors.rgb2hex(cmap(norm(value))) for value in gradient]         # Generation of green codes in order
    green=green[0:-2]


    ################################ PLOTTING ################################
    ##### LABELS #####
    labels={"A1ml":"[O2]<1ml",
            "V1ml":"[O2]<1ml",
            "A43ml":"[O2]<43ml",
            "V43ml":"[O2]<43ml"}
    
    ##### FIGURE #####
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
    fig.suptitle("AVERAGE FOR OXYGENE AND AGE CLASSES")
    fig.text(0.5, 0.92, file_title, ha='center',fontweight='bold')
    
    # --> 2nd axis: Parameter axis Legend (on the right) for the parameter
    axes2_0 = axes[0].twinx()
    axes2_1 = axes[1].twinx()
    
    # --> Stores the rows for each columns of parameter
    lines_params_0 = []                                                        # --> Areas graph                                                      
    lines_params_1 = []                                                        # --> Volumes graph 
     
    # --> Type of parameters (Series or DataFrame) : Stores and plotes all the information for each column
    if isinstance(parameters, pd.core.series.Series):                                               
        line_param_0, = axes2_0.plot(parameters, marker=' ', color=green[0], label=parameters.name)
        lines_params_0.append(line_param_0)
        line_param_1, = axes2_1.plot(parameters, marker=' ', color=green[0], label=parameters.name)
        lines_params_1.append(line_param_1)
    else:
        for i, col in enumerate(parameters.columns):
            line_param_0, = axes2_0.plot(parameters[col], marker=' ', color=green[i], label=col)
            lines_params_0.append(line_param_0)
            line_param_1, = axes2_1.plot(parameters[col], marker=' ', color=green[i], label=col)
            lines_params_1.append(line_param_1)
    
    # --> Legend for the second axis off each graph
    if file_title=="ABONDANCE HERRING" or file_title=="ABONDANCE SPRAT":
        axes2_1.set_ylabel("Abundance")                                        # --> Areas graph  
        axes2_1.tick_params('y')
        axes2_0.set_ylabel("Abundance")                                        # --> Volumes graph   
        axes2_0.tick_params('y') 
    else :
        axes2_1.set_ylabel(axe_title)                                          # --> Areas graph   
        axes2_1.tick_params('y')
        axes2_0.set_ylabel(axe_title)                                          # --> Volumes graph   
        axes2_0.tick_params('y')   
    
    axes2_0.tick_params(axis='y', labelsize=15)
    # --> Color, size and thickness of ordinate axes
    axes2_0.yaxis.label.set_color("green")
    axes2_0.yaxis.label.set_fontsize(15) 
    axes2_0.yaxis.label.set_fontweight('bold')  
    
    axes2_1.tick_params(axis='y', labelsize=15)
    # --> Color, size and thickness of ordinate axes
    axes2_1.yaxis.label.set_color("green")
    axes2_1.yaxis.label.set_fontsize(15) 
    axes2_1.yaxis.label.set_fontweight('bold')  
    
    # --> Legend of each parameters plot
    labels_params_common = [line.get_label() for line in lines_params_0]
    fig.legend(lines_params_0, labels_params_common, loc='lower center', prop={'size': 15}, ncol=5, bbox_to_anchor=(0.5, -0.01))
    
    # --> Plotting
    for k in [0,1]:
        if k ==0:
            ##### AREAS #####
            # --> Plot oxygen datas
            col="blue"
            line1, = axes[k].plot(oxy.iloc[:,0], marker=' ', color=colors.iloc[0, 0], label=labels[str(oxy.columns[0])])
            line2, = axes[k].plot(oxy.iloc[:,1], marker=' ', color=colors.iloc[0, 1], label=labels[str(oxy.columns[1])])
            axes[0].set_title('Evolution of the average area of hypoxic seafloor by years')
            axes[k].set_ylabel('Areas (km2)')
            
            # --> Plot all the Areas legend
            axes[k].legend(handles=[line1, line2], loc="upper right", prop={'size': 15})
        
        else:
            ##### VOLUMES #####
            # --> Plot oxygen datas
            col="red"
            line3, = axes[k].plot(oxy.iloc[:,2], marker=' ', color=colors.iloc[0, 2], label=labels[str(oxy.columns[2])])
            line4, = axes[k].plot(oxy.iloc[:,3], marker=' ', color=colors.iloc[0, 3], label=labels[str(oxy.columns[3])])
            axes[1].set_title('Evolution of the average volume of hypoxic seafloor by years')
            axes[k].set_ylabel('Volumes (km3)')
            # --> Plot all the Volumes legend
            axes[k].legend(handles=[line3, line4], loc="upper right", prop={'size': 15})
        
        # --> Abscissa
        axes[k].set_xlabel('Years')
       
        # --> Customizing tick labels font size
        axes[k].tick_params(axis='x', labelsize=15, rotation=45)
        axes[k].tick_params(axis='y', labelsize=15)
        
        # --> Size and thickness of abscissa axes
        axes[k].xaxis.label.set_fontsize(15) 
        axes[k].xaxis.label.set_fontweight('bold')  
        
        # --> Color, size and thickness of ordinate axes
        axes[k].yaxis.label.set_color(col)
        axes[k].yaxis.label.set_fontsize(15) 
        axes[k].yaxis.label.set_fontweight('bold')  
    
        
    ##### LEGEND #####    
    axes[0].text(0.02, 0.98, 'A', transform=axes[0].transAxes, fontsize=16, fontweight='bold', va='top')
    axes[1].text(0.02, 0.98, 'B', transform=axes[1].transAxes, fontsize=16, fontweight='bold', va='top')
    

    ##### FIGURE #####
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4,bottom=0.2)
    
    return()





###############################################################################
############################### ALL PARAMETERS ###############################
###############################################################################

"This function selects each parameters and uses the function PLOT_Evolution_Data_by_Average"
"for plot the evolution of area and volume of hypoxic zone and of one other paremeter by year"

def PLOT_Evolution_Data_by_Average_param(FILE_OXY_GLOBAL,file_parameters,file_title):
    
    # --> Selection of the parameters
    for k in range (len(file_parameters)):
        # --> Delete "[]" and devide character string from ":"
        parameters = file_parameters[k].strip("[]").split(":")
    
        # --> Conversion of the string caracters in integers
        parameters = [int(param)+4 for param in parameters]
        
        # --> Shows the evolution of the average of each parameters by years on FILE data
        PLOT_Evolution_Data_by_Average(FILE_OXY_GLOBAL,parameters,file_title)   
        
    return()





###############################################################################
################################### FIGURES ###################################
###############################################################################

############################### ABONDANCE SPRAT ###############################
# --> Title
ab_s_title="ABONDANCE SPRAT"

##### ALL AGE CLASSES #####
ab_s_parameters=["0:10"]
# --> Average by Years
PLOT_Evolution_Data_by_Average_param(AB_S_OXY_GLOBAL,ab_s_parameters,ab_s_title)


############################## ABONDANCE HERRING ##############################
# --> Title
ab_h_title="ABONDANCE HERRING"

##### ALL AGE CLASSES #####
ab_h_parameters=["0:10"]
# --> Average by Years
PLOT_Evolution_Data_by_Average_param(AB_H_OXY_GLOBAL,ab_h_parameters,ab_h_title)










###############################################################################
###############################################################################
##################### OXYGEN CONCENTRATION CLASSES & MAPS #####################
###############################################################################
###############################################################################


###############################################################################
############################## CALLING THE FILES ##############################
###############################################################################

##### PROBABILITIES #####
    # --> EM=Empirical
    # --> TH=Theorical

# --> Cod
PROBA_COD_EM = 'PROBA_COD_EM.csv'
chemin_complet = os.path.join(LOCATION_Data, PROBA_COD_EM)
PROBA_COD_EM=pd.read_csv(chemin_complet) 

PROBA_COD_TH = 'PROBA_COD_TH.csv'
chemin_complet = os.path.join(LOCATION_Data, PROBA_COD_TH)
PROBA_COD_TH=pd.read_csv(chemin_complet) 

# --> Flounder
PROBA_FLE_EM = 'PROBA_FLOUNDER_EM.csv'
chemin_complet = os.path.join(LOCATION_Data, PROBA_FLE_EM)
PROBA_FLE_EM=pd.read_csv(chemin_complet) 

PROBA_FLE_TH = 'PROBA_FLOUNDER_TH.csv'
chemin_complet = os.path.join(LOCATION_Data, PROBA_FLE_TH)
PROBA_FLE_TH=pd.read_csv(chemin_complet) 

# --> SPRAT
PROBA_SPR_EM = 'PROBA_SPRAT_EM.csv'
chemin_complet = os.path.join(LOCATION_Data, PROBA_SPR_EM)
PROBA_SPR_EM=pd.read_csv(chemin_complet) 

PROBA_SPR_TH = 'PROBA_SPRAT_TH.csv'
chemin_complet = os.path.join(LOCATION_Data, PROBA_SPR_TH)
PROBA_SPR_TH=pd.read_csv(chemin_complet) 

# --> HERRING
PROBA_HER_EM = 'PROBA_HERRING_EM.csv'
chemin_complet = os.path.join(LOCATION_Data, PROBA_HER_EM)
PROBA_HER_EM=pd.read_csv(chemin_complet) 

PROBA_HER_TH = 'PROBA_HERRING_TH.csv'
chemin_complet = os.path.join(LOCATION_Data, PROBA_HER_TH)
PROBA_HER_TH=pd.read_csv(chemin_complet) 


##### ICES #####
# --> COD
COD_ICES='COD_ICES.csv'
chemin_complet = os.path.join(LOCATION_Data, COD_ICES)
COD_ICES=pd.read_csv(chemin_complet) 
COD_ICES=list(COD_ICES["ICES"])

# --> FLOUNDER
FLE_ICES='FLOUNDER_ICES.csv'
chemin_complet = os.path.join(LOCATION_Data, FLE_ICES)
FLE_ICES=pd.read_csv(chemin_complet) 
FLE_ICES=list(FLE_ICES["ICES"])

# --> SPRAT
SPR_ICES='SPRAT_ICES.csv'
chemin_complet = os.path.join(LOCATION_Data, SPR_ICES)
SPR_ICES=pd.read_csv(chemin_complet) 
SPR_ICES=list(SPR_ICES["ICES"])

# --> HERRING
HER_ICES='HERRING_ICES.csv'
chemin_complet = os.path.join(LOCATION_Data, HER_ICES)
HER_ICES=pd.read_csv(chemin_complet) 
HER_ICES=list(HER_ICES["ICES"])


##### OXYGEN MAP #####
OXY_M='OXY_MAP.csv'
chemin_complet = os.path.join(LOCATION_Data, OXY_M)
OXY_M=pd.read_csv(chemin_complet) 



############################## SET UP THE INDEX ##############################
# --> Cod
PROBA_COD_EM = INDEX_GLOBAL(PROBA_COD_EM)
PROBA_COD_TH = INDEX_GLOBAL(PROBA_COD_TH)
# --> Flounder
PROBA_FLE_EM = INDEX_GLOBAL(PROBA_FLE_EM)
PROBA_FLE_TH = INDEX_GLOBAL(PROBA_FLE_TH)
# --> Sprat
PROBA_SPR_EM = INDEX_GLOBAL(PROBA_SPR_EM)
PROBA_SPR_TH = INDEX_GLOBAL(PROBA_SPR_TH)
# --> Herring
PROBA_HER_EM = INDEX_GLOBAL(PROBA_HER_EM)
PROBA_HER_TH = INDEX_GLOBAL(PROBA_HER_TH)



################################ DATA STORAGE ################################
# --> Stores all EMPIRICAL probability of all species
PROBA_EM={"COD": deepcopy(PROBA_COD_EM) , "FLOUNDER" : deepcopy(PROBA_FLE_EM) ,
       "SPRAT" : deepcopy(PROBA_SPR_EM) , "HERRING" : deepcopy(PROBA_HER_EM)}
# --> Stores all THEORICAL probability of all species
PROBA_TH={"COD": deepcopy(PROBA_COD_TH) , "FLOUNDER" : deepcopy(PROBA_FLE_TH) ,
       "SPRAT" : deepcopy(PROBA_SPR_TH) , "HERRING" : deepcopy(PROBA_HER_TH)}



########################### OPTIMUM YEAR SELECTION ###########################
OXY_opti={"Min":OXY_GLOBAL['A1ml'].idxmin(),
           "Max":OXY_GLOBAL['A43ml'].idxmax()}





###############################################################################
############################### ICES CONVERSION ###############################
###############################################################################

" This function uses the ICES code to calculate the latitude and the longitude "
" of the sample following the ices.dk code : " 
" https://www.ices.dk/data/maps/Pages/ICES-statistical-rectangles.aspx "

def Extract(OXY):
    
    # Extraction of part of the ICES number that codes for the latitude and the longitude
    OXY["Lat"]=OXY["ICES"].str.slice(0, 2)
    OXY["Long"]=OXY["ICES"].str.slice(2, 3)
    OXY["Long2"]=OXY["ICES"].str.slice(3, 4)
    OXY["Long2"]=OXY["Long2"].astype(float)
    
    ##### LATITUDE #####
    for k in range(len(OXY)):
        Lat=int(OXY["Lat"].iloc[k])
        LAT=float(36+int(Lat*0.50))
        if (Lat*0.50)%1==0 and Lat!=1:
            LAT=float(LAT+0.5)
            
        OXY["Lat"].iloc[k]=LAT
        
    ##### LONGITUDE #####
    LONG=pd.DataFrame()
    LONG["ICES"]=("A","B","C","D","E","F","G","H","J","K","L","M")
    LONG["Coor"]=(40,30,20,10,0,0,10,20,30,40,50,60)  # revoir
    # --> Association with letter and tens
    mapping_dict = dict(zip(LONG["ICES"], LONG["Coor"]))
    OXY["Long"] = OXY["Long"].map(mapping_dict)
    # --> Remonving nan values (due the I letter in the ICES)
    nan = OXY.isna().any(axis=1)
    OXY = OXY[~nan]
    # --> Calculation of longitude values
    OXY["Long"]=OXY["Long"]+OXY["Long2"]

    del(OXY["Long2"])

    ##### RETURN #####
    return(OXY)





###############################################################################
########## EXTRACTING THE DATA & OXYGEN COCENTRATION CLASSES SET UP ##########
###############################################################################


# --> New file 
OXY_MAP=deepcopy(OXY_M)
# --> Set-up the classes
    # --> 0 if >43ml
    # --> 1 if <43ml
    # --> 2 if <1ml
OXY_MAP.loc[OXY_MAP["A43ml"] != 0, "A43ml"] = 1
OXY_MAP.loc[OXY_MAP["V43ml"] != 0, "V43ml"] = 1
OXY_MAP.loc[OXY_MAP["A1ml"] != 0, "A1ml"] = 2
OXY_MAP.loc[OXY_MAP["V1ml"] != 0, "V1ml"] = 2
OXY_MAP["Area"]=0
OXY_MAP["Volume"]=0
# --> For each location, selection of the highest value of the classes for each ICES of each years
for location in range(len(OXY_MAP)):
    OXY_MAP["Area"].iloc[location]=max(OXY_MAP["A43ml"].iloc[location],OXY_MAP["A1ml"].iloc[location])
    OXY_MAP["Volume"].iloc[location]=max(OXY_MAP["V43ml"].iloc[location],OXY_MAP["V1ml"].iloc[location])
    





###############################################################################
########################## OXYGEN CONCENTRATION MAPS ##########################
###############################################################################

" This function uses the ICES code to calculate (for the year considered) the "
" latitude and the longitude of the sample following the ices.dk code : " 
" https://www.ices.dk/data/maps/Pages/ICES-statistical-rectangles.aspx "

def PLOT_MAP_OVERLEAF(FILE,year,title):
    
    ######################### SET UP THE INFORMATIONS ########################
    # --> New file
    OXY=deepcopy(FILE)
    # --> Sorted all the values in area and volume columns
    values=sorted(list(set(OXY["Area"])))  
    # --> Stores the minimal and maximal values for each parameter
    optimals = {"Min":min(OXY["Area"]),"Max":max(OXY["Area"])}
    
    ##### VALUES #####
    # --> Selection for values for the year considered
    plot_values = OXY.loc[OXY["Year"] == year]
    # --> Oxygen values
    plot_oxy2 = plot_values.pivot_table(index='Lat', columns='Long',
                                       values="Area", aggfunc='mean')  
    # --> Latitude and Longitude values
    plot_lat=list(plot_oxy2.index)
    plot_lon=list(plot_oxy2.columns)
    # --> Map borders
    Lat_max=max(plot_lat)
    Lat_min=min(plot_lat)
    Lon_max=max(plot_lon)
    Lon_min=min(plot_lon)
    
    ####################### OXYGEN CONCENTRATIONS GRAPHS ######################
    ##### SET UP THE FIGURE #####
    plt.figure(figsize=(15, 10))
    # --> Title
    plt.suptitle("Distribution of Oxygen Concentration in the Baltic sea in "+str(title)+" sampling zone",fontsize=15)
    plt.title(str(year),fontsize=20, fontweight='bold')
    # --> Plot
    plt.contourf(plot_lon,plot_lat,plot_oxy2,cmap='coolwarm',
                   vmin=optimals["Min"], vmax=optimals["Max"], levels=2)
    plt.grid(True)
    # --> Axes title and Legend
    plt.xlabel('Longitude',fontsize=15,fontweight='bold')
    plt.ylabel('Latitude',fontsize=15,fontweight='bold')
    plt.xlim(Lon_min, Lon_max)
    plt.ylim(Lat_min, Lat_max)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if year==OXY_opti["Min"]:
        plt.text(0.02, 0.98, 'A', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')
    if year==OXY_opti["Max"]:
        plt.text(0.02, 0.98, 'B', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

    ##### COLORBAR #####
    # --> Calculation of ticks position as the middle of the color intervals
    tick_positions = np.linspace(0.4, len(values) - 1, len(values))
    # --> Definition of de colorbar ticks with the calculated positions
    cbar = plt.colorbar(label='Oxygen concentration', ticks=tick_positions)
    cbar.set_ticklabels(['[O2]>43ml', '1ml\u2264[O2]\u226443ml', '[O2]<1ml'],fontsize=12)
    cbar.ax.yaxis.label.set_size(15)
    cbar.ax.yaxis.label.set_weight('bold')

    ################################# RESULTS ################################
    
    return ()





###############################################################################
################################### FIGURES ###################################
###############################################################################
##### GLOBAL #####
PLOT_MAP_OVERLEAF(OXY_MAP,OXY_opti["Min"],"Oxygen")
PLOT_MAP_OVERLEAF(OXY_MAP,OXY_opti["Max"],"Oxygen")


##### COD & FLE#####
# --> Oxygen values for cod sample zone
COD_MAP = deepcopy (OXY_MAP)
COD_MAP = COD_MAP[COD_MAP.isin(COD_ICES).any(axis=1)]
COD_MAP = Extract(COD_MAP)

# --> Figures for optimum years
PLOT_MAP_OVERLEAF(COD_MAP,OXY_opti["Min"], "Cod & Flounder")
PLOT_MAP_OVERLEAF(COD_MAP,OXY_opti["Max"], "Cod & Flounder")


##### SPRAT & HERRING #####
# --> Oxygen values for sprat sample zone
SPR_MAP = deepcopy (OXY_MAP)
SPR_MAP = SPR_MAP[SPR_MAP.isin(SPR_ICES).any(axis=1)]
SPR_MAP = Extract(SPR_MAP)

# --> Figures for optimum years
PLOT_MAP_OVERLEAF(SPR_MAP,OXY_opti["Min"], "Herring & Sprat")
PLOT_MAP_OVERLEAF(SPR_MAP,OXY_opti["Max"], "Herring & Sprat")










###############################################################################
###############################################################################
################################ PROBABILITIES ################################
###############################################################################
###############################################################################


###############################################################################
############################## PROBABILITY BARS ##############################
###############################################################################

" This function shows the temporal series of probabilities for the specie chosen, "
" for the parameter (AREA or VOLUME) chosen. "

def PLOT_PROBABILITY_OVERLEAF(PROBA_EM , PROBA_TH , specie , year) :
    
    ######################### SET UP THE INFORMATIONS ########################
    # --> Selection AREA datas
    params=PROBA_EM[specie].columns[0:6]
    params=sorted(list(params))
    params=params[0:3]
    # --> New labels
    labels=['[O2]<1ml','1ml\u2264[O2]\u226443ml','43ml<[O2]']

    ##### DATAS #####
    if year not in PROBA_EM[specie].index or year not in PROBA_TH[specie].index:
        return ("There is no data for the year "+str(year)+" for the "+specie+".")
    if PROBA_EM[specie].loc[year].isna().any() or PROBA_TH[specie].loc[year].isna().any():
        return ("There is no data for the year "+str(year)+" for the "+specie+".")

    # --> Stores EMPIRICAL probability of the specie and the year considered
    FILE_EM=deepcopy(PROBA_EM[specie].loc[year])
    # --> Stores THEORICAL probability of the specie and the year considered
    FILE_TH=deepcopy(PROBA_TH[specie].loc[year])
    
    ############################ PROBABILITY GRAPHS ###########################   
    ##### SET UP THE FIGURE #####
    plt.figure(figsize=(15,9))
    plt.suptitle("Empirical and theorical expectation to find an individual in each Oxygen cencatrion zone for "+str(specie), fontsize=15)
    plt.title(year,fontsize=20, fontweight='bold')
    plt.xlabel("")
    plt.ylabel("Probability",fontsize=15,fontweight="bold")
    plt.ylim(0, 1.05)
    plt.yticks(fontsize=15)
    # --> Set up the position and labels of x-axis
    plt.xticks(range(3),labels,fontsize=15)
    # --> Width of the bar of the histogram 
    width_bar = 0.35

    ##### BARS #####
    # --> Association parameters with colors (same colors than on the maps)
    colors={str(params[0]):['#b70d28','#666666'],
           str(params[1]):['#f2cbb7','#999999'],
           str(params[2]):['#5977e3', '#CCCCCC']}
    
    # --> Correct labels for the graphs
    labels={str(params[0]):['[O2]<1ml EM','[O2]<1ml TH'],
            str(params[1]):['1ml\u2264[O2]\u226443ml EM','1ml\u2264[O2]\u226443ml TH'],
            str(params[2]):['43ml<[O2] EM','43ml<[O2] TH']}
    
    # --> Bar position on the x-axis
    position = np.arange(len(params))
    for i in position:
        # --> Selection of parameter
        param=params[i]
        
        ##### EMPIRICAL #####
        value1=FILE_EM.loc[param]
        # --> Plot of EMPIRICAL bars in RED
        plt.bar(position[i] - width_bar/2, value1, width_bar, color=colors[param][0],
                label=labels[param][0])
        
        ##### THEORICAL #####
        value2=FILE_TH.loc[param]
        # --> Plot of THEORICAL bars in BLUE
        plt.bar(position[i] + width_bar/2, value2, width_bar, color=colors[param][1],
                label=labels[param][1])
    
                
    ##### LEGEND #####
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.04),ncol=3,fontsize=12)
    if specie=="COD" or specie=="FLOUNDER":
        if year==OXY_opti["Max"]:
            plt.text(0.02, 0.98, 'C', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')
    if specie=="HERRING" or specie=="SPRAT":
        if year==OXY_opti["Min"]:
            plt.text(0.02, 0.98, 'C', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')
        if year==OXY_opti["Max"]:
            plt.text(0.02, 0.98, 'D', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')
    plt.show()
    
    ################################# RESULTS ################################
    
    return ()





################################### FIGURES ###################################
##### COD #####
for year in OXY_opti:
    PLOT_PROBABILITY_OVERLEAF(PROBA_EM , PROBA_TH , "COD" , OXY_opti[year])
    
##### FLOUNDER #####
for year in OXY_opti:
    PLOT_PROBABILITY_OVERLEAF(PROBA_EM , PROBA_TH , "FLOUNDER" , OXY_opti[year])
    
##### HERRING #####
for year in OXY_opti:
    PLOT_PROBABILITY_OVERLEAF(PROBA_EM , PROBA_TH , "HERRING" , OXY_opti[year])
    
##### SPRAT #####
for year in OXY_opti:
    PLOT_PROBABILITY_OVERLEAF(PROBA_EM , PROBA_TH , "SPRAT" , OXY_opti[year])  





###############################################################################
######################### PROBABILITY TEMPORAL SERIES #########################
###############################################################################

" This function shows the temporal series of probabilities for the specie selected, "
" for the parameter (AREA or VOLUME) chosen. "

def PLOT_PROBABILITY_EVOLUTION_SPECIE(specie , PROBA_PLOT_EM , PROBA_PLOT_TH , PARAM , OXY_opti):
    
    ######################### SET UP THE INFORMATIONS #########################
    # --> Datas
    SPECIE_EM=PROBA_PLOT_EM[specie]
    SPECIE_TH=PROBA_PLOT_TH[specie]
    
    # --> DataFrame of references for each specie
    REF=PROBA_PLOT_EM[specie]
    # --> Creation of range of years since the first year of sample to the last
    years=[i for i in range(REF.index[0],REF.index[-1]+1)]
    # --> Stores even years
    if specie!="FLOUNDER":
        years = [paire for paire in years if paire % 2 == 0]
    # --> Optimum years
    year_min=OXY_opti["Min"]
    year_max=OXY_opti["Max"]
    # --> Order parameters (Area then Volume)
    params=PROBA_PLOT_EM["COD"].columns[0:6]
    params=sorted(list(params))

    # --> Area
    if PARAM=="AREA":
       params=params[0:3]
    # --> Volume
    else:
       params=params[3:6]
    # --> Association parameters with colors (same colors than on the maps)
    colors={str(params[0]):['#b70d28','#666666'],
           str(params[1]):['#f2cbb7','#999999'],
           str(params[2]):['#5977e3', '#CCCCCC']}
    # --> Correct labels for the graphs
    labels={str(params[0]):['[O2]<1ml EM','[O2]<1ml TH'],
            str(params[1]):['1ml\u2264[O2]\u226443ml EM','1ml\u2264[O2]\u226443ml TH'],
            str(params[2]):['43ml<[O2] EM','43ml<[O2] TH']}
    
    
    ################################# FIGURES #################################
    ##### SET UP THE FIGURE #####
    plt.figure(figsize=(15,9))
    plt.title("Time series of the empirical and theorical probability for the three Oxygen Concentration for "+str(specie), fontsize=15)
    
    ##### LABEL & AXES #####
    plt.xlabel("Years",fontsize=15,fontweight='bold')
    plt.ylabel("Probability",fontsize=15,fontweight='bold')
    plt.xticks(years+ [year_min,year_max], fontsize=15)
    plt.yticks(fontsize=15)
    plt.tick_params(axis='x', rotation=45)
    plt.ylim(-0.05, 1.05)
    plt.xlim(REF.index[0]-0.5,REF.index[-1]+0.5)
    
    ##### PLOT #####
    # --> Probability datas
    for param in params:
        plt.plot(SPECIE_EM.index,SPECIE_EM[param],label=labels[param][0],color=colors[param][0])   
        plt.plot(SPECIE_EM.index,SPECIE_EM[param],'.',color=colors[param][0])   
        plt.plot(SPECIE_TH.index,SPECIE_TH[param],label=labels[param][1],color=colors[param][1])
        plt.plot(SPECIE_TH.index,SPECIE_TH[param],'.',color=colors[param][1])    
    # --> Optimum years
    plt.axvline(x=year_min, color='green', linestyle='--',label=str(year_min)+":lowest Area Hypoxic zone") 
    plt.axvline(x=year_max, color='green', linestyle='--',label=str(year_max)+":highest Area Hypoxic zone")

    ##### LEGEND #####
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.135), fancybox=True, ncol=4, fontsize=12)
    plt.subplots_adjust(bottom=0.22) 
    if specie=="COD" or specie=="FLOUNDER":
        plt.text(0.02, 0.98, 'D', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')
    if specie=="HERRING" or specie=="SPRAT":
        plt.text(0.02, 0.98, 'E', transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')
    
    
    ################################# RESULTS #################################
    
    return()





################################### FIGURES ###################################
##### COD #####
PLOT_PROBABILITY_EVOLUTION_SPECIE("COD" , PROBA_EM , PROBA_TH , "AREA" , OXY_opti)

##### FLOUNDER #####
PLOT_PROBABILITY_EVOLUTION_SPECIE("FLOUNDER" , PROBA_EM , PROBA_TH , "AREA" , OXY_opti)

##### SPRAT #####
PLOT_PROBABILITY_EVOLUTION_SPECIE("SPRAT" , PROBA_EM , PROBA_TH , "AREA" , OXY_opti)

##### HERRING #####
PLOT_PROBABILITY_EVOLUTION_SPECIE("HERRING" , PROBA_EM , PROBA_TH , "AREA" , OXY_opti)