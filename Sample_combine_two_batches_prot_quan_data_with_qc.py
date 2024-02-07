# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 21:34:24 2021

@author: schepyal and adasgupt

modified on Wed Feb  7 15:32:44 2024
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd, re, numpy as np, os, sys
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr

#########################
 # Small module functions
 ########################
 
def getParams(paramFile):
    parameters = dict()
    with open(paramFile, 'r') as file:
        for line in file:
            if re.search(r'^#', line) or re.search(r'^\s', line):
                continue
            line = re.sub(r'#.*', '', line)  # Remove comments (start from '#')
            line = re.sub(r'\s*', '', line)  # Remove all whitespaces
            # Exception for "feature_files" parameter
            if "feature_files" in parameters and line.endswith("feature"):
                parameters["feature_files"].append(line)
            else:
                #print("line = %s" %line)
                key = line.split('=')[0]
                val = line.split('=')[1]
                if key == "feature_files":
                    parameters[key] = [val]
                else:
                    parameters[key] = val
    return parameters

def QC_degpat(subDf2_, x,index,i, replicate_colNames):
    if x[index] == 4:
        subDf2_.iloc[i,subDf2_.columns.get_loc(replicate_colNames[0])] = np.nan 
    elif x[index] == 8:
        subDf2_.iloc[i,subDf2_.columns.get_loc(replicate_colNames[1])] = np.nan
    elif x[index] == 16:
        subDf2_.iloc[i,subDf2_.columns.get_loc(replicate_colNames[2])] = np.nan
    elif x[index] == 32:
        subDf2_.iloc[i, subDf2_.columns.get_loc(replicate_colNames[3])] = np.nan
    return subDf2_
def func(x, a, b):
    return np.exp(-b * x)
           
def outlier_degPattern(subDf2_,set_1, set_2, delete_3data_notfollowDegRate):
    for i in range(len(subDf2_)):
        for k in range(1,3):
            replicate_colNames = locals()["set_" + str(k)][1:5]
            y = subDf2_[replicate_colNames].iloc[i]
            y = pd.concat([pd.Series([1]), y]); 
            y1 = np.array(y.values.tolist()); 
            y1 = y1[np.where(y.notnull())];
            x = np.array([0, 4, 8, 16, 32])
            x = x[np.where(y.notnull())]; 
            while any(earlier < later for earlier, later in zip(y1, y1[1:])):
                if len(y1) == 3:
                    if (delete_3data_notfollowDegRate == 0):
                        popt, pcov = curve_fit(func, x, y1)
                        error = abs(y1 - func(x, *popt)); 
                        maxerror_index = np.where(error == error.max())
                        subDf2_ = QC_degpat(subDf2_,x,maxerror_index,i,replicate_colNames)
                    else:
                        subDf2_.iloc[i,[subDf2_.columns.get_loc(replicate_colNames[0]), subDf2_.columns.get_loc(replicate_colNames[1]), subDf2_.columns.get_loc(replicate_colNames[2]), subDf2_.columns.get_loc(replicate_colNames[3])]] = np.nan

                    y = subDf2_[replicate_colNames].iloc[i]
                    y = pd.concat([pd.Series([1]), y]); 
                    y1 = np.array(y.values.tolist()); 
                    y1 = y1[np.where(y.notnull())];
                    x = np.array([0, 4, 8, 16, 32])
                    x = x[np.where(y.notnull())];    
                elif len(y1) > 3:
                    diff_y1 = np.diff(y1); 
                    ind = np.argmax(diff_y1, axis=0)+1; 
                    if (ind != len(y1)-1):
                        if (y1[ind] > y1[ind-1]) and (y1[ind] > y1[ind+1]) and (y1[ind+1] < y1[ind-1]):
                            subDf2_ = QC_degpat(subDf2_,x,ind,i, replicate_colNames) 
                        elif (y1[ind] > y1[ind-1]) and (y1[ind] > y1[ind+1]) and (y1[ind+1] > y1[ind-1]):
                            if ind == 1:
                                subDf2_ = QC_degpat(subDf2_,x,ind,i,replicate_colNames)
                            else:
                                subDf2_ = QC_degpat(subDf2_,x,ind-1,i,replicate_colNames)
                        elif (y1[ind] > y1[ind-1]) :
                           popt, pcov = curve_fit(func, x, y1)
                           error = abs(y1 - func(x, *popt)); 
                           maxerror_index = np.where(error == error.max())
                           subDf2_ = QC_degpat(subDf2_,x,maxerror_index,i,replicate_colNames)
                    else:
                        if (y1[ind] > y1[ind-1]) and (y1[ind] > y1[ind-2]):
                            subDf2_ = QC_degpat(subDf2_,x,ind,i,replicate_colNames)
                        elif (y1[ind] > y1[ind-1]) :
                            popt, pcov = curve_fit(func, x, y1)
                            error = abs(y1 - func(x, *popt)); 
                            maxerror_index = np.where(error == error.max())
                            subDf2_ = QC_degpat(subDf2_,x,maxerror_index,i,replicate_colNames)
                    y = subDf2_[replicate_colNames].iloc[i]
                    y = pd.concat([pd.Series([1]), y]); 
                    y1 = np.array(y.values.tolist()); 
                    y1 = y1[np.where(y.notnull())];
                    x = np.array([0, 4, 8, 16, 32])
                    x = x[np.where(y.notnull())];
                else:
                    subDf2_.iloc[i,[subDf2_.columns.get_loc(replicate_colNames[0]), subDf2_.columns.get_loc(replicate_colNames[1]), subDf2_.columns.get_loc(replicate_colNames[2]), subDf2_.columns.get_loc(replicate_colNames[3])]] = np.nan
                
                    y = subDf2_[replicate_colNames].iloc[i]
                    y = pd.concat([pd.Series([1]), y]); 
                    y1 = np.array(y.values.tolist());
                    y1 = y1[np.where(y.notnull())];
                    x = np.array([0, 4, 8, 16, 32])
                    x = x[np.where(y.notnull())];
    return subDf2_

# In[1]: 
if __name__ == "__main__":
    
    print(" Program for combining two turnover data from two batches and summarize to genotype level \n")
    
    # Read parameter file
    paramFile = sys.argv[1]
    params = getParams(paramFile)
    print("parameters used in this program\n")
    print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in params.items()) + "}")
    
    input_folders = [k for k,v in params.items() if k.startswith("batch")]
    reporters = params["tmt_reporters_used"].split(";")
    free_Lys_ratio =params["free_Lys_ratio"].split(",")# [1, .745, .637, .497, .35];
    free_Lys_ratio = [float(i) for i in free_Lys_ratio]
    WT1 = params["WT1"].split(",")
    WT2 = params["WT2"].split(",")
    WT3 = params["WT3"].split(",")
    FAD1 = params["FAD1"].split(",")
    FAD2 = params["FAD2"].split(",")
    FAD3 = params["FAD3"].split(",")
    
    if os.path.exists(os.path.join(os.getcwd(), params["output_folder"])):
        i = 1
        while os.path.exists(os.path.join(os.getcwd(),  params["output_folder"]+'_%s' %i)):
            i += 1
        final_directory = os.path.join(os.getcwd(),  params["output_folder"]+'_%s' %i)
        os.makedirs(final_directory, exist_ok=True)
    else: 
        final_directory = os.path.join(os.getcwd(),  params["output_folder"])
        os.makedirs(os.path.join(os.getcwd(), params["output_folder"]))

    for j  in range(0, len(input_folders)):
        #j=0
        #####################################
        ## Read input data (uni_pep_quant data)
        #####################################
        all_protein = pd.read_csv(os.path.join(params[input_folders[j]], 'all_protein_turnover.txt'),  sep="\t", skiprows=1,index_col=0)
        uni_protein = pd.read_csv(os.path.join(params[input_folders[j]], 'uni_protein_turnover.txt'),  sep="\t", skiprows=1,index_col=0)
        
        all_peptide     = pd.read_csv(os.path.join(params[input_folders[j]], 'peptide_turnover.txt'),  sep="\t", skiprows=1,index_col=0)
        if 'Protein' in all_peptide.columns:
            all_peptide["key"] = all_peptide["Peptide"] + "_" + all_peptide["Protein"]
            all_peptide.set_index('key', inplace = True)
            
        all_peptide = all_peptide.groupby(all_peptide.index).first()
        if j == 0:
            all_protein             = all_protein.add_prefix("b"+str(j+1)+"_")
            all_protein_all_batches = all_protein.copy()
            all_peptide             = all_peptide.add_prefix("b"+str(j+1)+"_")
            all_peptide_all_batches = all_peptide.copy()
            
            uni_protein             = uni_protein.add_prefix("b"+str(j+1)+"_")
            uni_protein_all_batches = uni_protein.copy()
        else:
            all_protein                 = all_protein.add_prefix("b"+str(j+1)+"_")
            all_protein_all_batches     = pd.concat([all_protein_all_batches, all_protein] , axis = 1, join = "outer")
            all_peptide                 = all_peptide.add_prefix("b"+str(j+1)+"_")
            all_peptide_all_batches     = pd.concat([all_peptide_all_batches, all_peptide] , axis = 1, join = "outer")
            uni_protein                 = uni_protein.add_prefix("b"+str(j+1)+"_")
            uni_protein_all_batches     = pd.concat([uni_protein_all_batches, uni_protein] , axis = 1, join = "outer")
            
    # In[1]: 
    ####################################### 
    # preparing the whole data to for print 
    #######################################
    
   
    if 'b1_normalized_status' in all_peptide_all_batches.columns:
        all_peptide_all_batches['b1_Peptide'] = all_peptide_all_batches.apply(lambda row: row['b2_Peptide'] if pd.isnull(row['b1_Peptide']) else row['b1_Peptide'],axis=1)
        all_peptide_all_batches['b1_GN'] = all_peptide_all_batches.apply(lambda row: row['b2_GN'] if pd.isnull(row['b1_GN']) else row['b1_GN'],axis=1)
        all_peptide_all_batches['b1_Protein Description'] = all_peptide_all_batches.apply(lambda row: row['b2_Protein Description'] if pd.isnull(row['b1_Protein Description']) else row['b1_Protein Description'],axis=1)
        all_peptide_all_batches['b1_Protein Accession #'] = all_peptide_all_batches.apply(lambda row: row['b2_Protein Accession #'] if pd.isnull(row['b1_Protein Accession #']) else row['b1_Protein Accession #'],axis=1)
        all_peptide_all_batches['b1_Protein Group#'] = all_peptide_all_batches.apply(lambda row: row['b2_Protein Group#'] if pd.isnull(row['b1_Protein Group#']) else row['b1_Protein Group#'],axis=1)
        all_peptide_all_batches['b1_normalized_status'] = all_peptide_all_batches.apply(lambda row: row['b2_normalized_status'] if pd.isnull(row['b1_normalized_status']) else row['b1_normalized_status'],axis=1)
        all_peptide_all_batches = all_peptide_all_batches.drop(['b2_Peptide', 'b2_GN','b2_Protein Description','b2_normalized_status', 'b2_Protein Accession #', 'b2_Protein Group#'], axis=1)
        all_peptide_all_batches.rename(columns={'b1_Peptide': 'Peptide','b1_GN': 'GN','b1_Protein Description':'Protein Description', 'b1_normalized_status':'normalized_status', 'b1_Protein Accession #': 'Protein Accession','b1_Protein Group#': 'Protein Group#'},inplace=True)
        all_peptide_all_batches.set_index('Peptide', inplace = True)
   
    # Print Peptide data
    all_peptide_all_batches['WT_day0'] = (all_peptide_all_batches[[WT1[0], WT2[0], WT3[0]]]).mean(axis=1)
    all_peptide_all_batches['WT_day4'] = (all_peptide_all_batches[[WT1[1], WT2[1], WT3[1]]]).mean(axis=1)
    all_peptide_all_batches['WT_day8'] = (all_peptide_all_batches[[WT1[2], WT2[2], WT3[2]]]).mean(axis=1)
    all_peptide_all_batches['WT_day16'] = (all_peptide_all_batches[[WT1[3], WT2[3], WT3[3]]]).mean(axis=1)
    all_peptide_all_batches['WT_day32'] = (all_peptide_all_batches[[WT1[4], WT2[4], WT3[4]]]).mean(axis=1)
    
    all_peptide_all_batches['FAD_day0'] = (all_peptide_all_batches[[FAD1[0], FAD2[0], FAD3[0]]]).mean(axis=1)
    all_peptide_all_batches['FAD_day4'] = (all_peptide_all_batches[[FAD1[1], FAD2[1], FAD3[1]]]).mean(axis=1)
    all_peptide_all_batches['FAD_day8'] = (all_peptide_all_batches[[FAD1[2], FAD2[2], FAD3[2]]]).mean(axis=1)
    all_peptide_all_batches['FAD_day16'] = (all_peptide_all_batches[[FAD1[3], FAD2[3], FAD3[3]]]).mean(axis=1)
    all_peptide_all_batches['FAD_day32'] = (all_peptide_all_batches[[FAD1[4], FAD2[4], FAD3[4]]]).mean(axis=1)
    
   # Print Protein data
    all_protein_all_batches['b1_GN'] = all_protein_all_batches.apply(lambda row: row['b2_GN'] if pd.isnull(row['b1_GN']) else row['b1_GN'],axis=1)
    all_protein_all_batches['b1_Protein Description'] = all_protein_all_batches.apply(lambda row: row['b2_Protein Description'] if pd.isnull(row['b1_Protein Description']) else row['b1_Protein Description'],axis=1)
    all_protein_all_batches['b1_Protein Group#'] = all_protein_all_batches.apply(lambda row: row['b2_Protein Group#'] if pd.isnull(row['b1_Protein Group#']) else row['b1_Protein Group#'],axis=1)
    if 'b1_normalized_status' in all_protein_all_batches.columns:
        all_protein_all_batches['b1_normalized_status'] = all_protein_all_batches.apply(lambda row: row['b2_normalized_status'] if pd.isnull(row['b1_normalized_status']) else row['b1_normalized_status'],axis=1)
        all_protein_all_batches = all_protein_all_batches.drop(['b2_GN','b2_Protein Description','b2_normalized_status', 'b2_Protein Group#'], axis=1)
        all_protein_all_batches.rename(columns={'b1_GN': 'GN','b1_Protein Description':'Protein Description', 'b1_normalized_status':'normalized_status', 'b1_Protein Group#': 'Protein Group'},inplace=True)
    else:
        all_protein_all_batches = all_protein_all_batches.drop(['b2_GN','b2_Protein Description', 'b2_Protein Group#'], axis=1)
        all_protein_all_batches.rename(columns={'b1_GN': 'GN','b1_Protein Description':'Protein Description', 'b1_Protein Group#': 'Protein Group'},inplace=True)

    all_protein_all_batches['WT_day0'] = (all_protein_all_batches[[WT1[0], WT2[0], WT3[0]]]).mean(axis=1)
    all_protein_all_batches['WT_day4'] = (all_protein_all_batches[[WT1[1], WT2[1], WT3[1]]]).mean(axis=1)
    all_protein_all_batches['WT_day8'] = (all_protein_all_batches[[WT1[2], WT2[2], WT3[2]]]).mean(axis=1)
    all_protein_all_batches['WT_day16'] = (all_protein_all_batches[[WT1[3], WT2[3], WT3[3]]]).mean(axis=1)
    all_protein_all_batches['WT_day32'] = (all_protein_all_batches[[WT1[4], WT2[4], WT3[4]]]).mean(axis=1)
    
    all_protein_all_batches['FAD_day0'] = (all_protein_all_batches[[FAD1[0], FAD2[0], FAD3[0]]]).mean(axis=1)
    all_protein_all_batches['FAD_day4'] = (all_protein_all_batches[[FAD1[1], FAD2[1], FAD3[1]]]).mean(axis=1)
    all_protein_all_batches['FAD_day8'] = (all_protein_all_batches[[FAD1[2], FAD2[2], FAD3[2]]]).mean(axis=1)
    all_protein_all_batches['FAD_day16'] = (all_protein_all_batches[[FAD1[3], FAD2[3], FAD3[3]]]).mean(axis=1)
    all_protein_all_batches['FAD_day32'] = (all_protein_all_batches[[FAD1[4], FAD2[4], FAD3[4]]]).mean(axis=1)
    
    FAD = ['FAD_day0', 'FAD_day4', 'FAD_day8', 'FAD_day16', 'FAD_day32']
    WT = ['WT_day0', 'WT_day4', 'WT_day8', 'WT_day16', 'WT_day32']
    if params["free_Lys_outlier"] == "1": # Remove data points (at Peptide level) that are faster  than Lys dyamics
        for i in range(1, len(free_Lys_ratio)):
            all_protein_all_batches[all_protein_all_batches[[FAD[i], WT[i]]] <= free_Lys_ratio[i]] = np.nan
            all_peptide_all_batches[all_peptide_all_batches[[FAD[i], WT[i]]] <= free_Lys_ratio[i]] = np.nan
    if params["free_Lys_outlier"] == "2": # Remove data points (at Peptide level) that are faster  than Lys dyamics
        for i in range(1, len(free_Lys_ratio)):
            for j in range (0, len(all_protein_all_batches)):
                if all_protein_all_batches.iloc[j,   all_protein_all_batches.columns.get_loc(FAD[i])] <= free_Lys_ratio[i]:
                    all_protein_all_batches.iloc[j,  all_protein_all_batches.columns.get_loc(FAD[i])]= free_Lys_ratio[i]
                if all_protein_all_batches.iloc[j,   all_protein_all_batches.columns.get_loc(WT[i])] <= free_Lys_ratio[i]:
                    all_protein_all_batches.iloc[j,  all_protein_all_batches.columns.get_loc(WT[i])]= free_Lys_ratio[i]
            for j in range (0, len(all_peptide_all_batches)):
                if  all_peptide_all_batches.iloc[j,  all_peptide_all_batches.columns.get_loc(FAD[i])] <= free_Lys_ratio[i]:
                    all_peptide_all_batches.iloc[j,  all_peptide_all_batches.columns.get_loc(FAD[i])]= free_Lys_ratio[i]
                if  all_peptide_all_batches.iloc[j,  all_peptide_all_batches.columns.get_loc(WT[i])] <= free_Lys_ratio[i]:
                    all_peptide_all_batches.iloc[j,  all_peptide_all_batches.columns.get_loc(WT[i])]= free_Lys_ratio[i]    
    if params["degradation_pattern_outlier"] == "1": # Remove data points (at Peptide level) that do not follow degredation pattern
        delete_3data_notfollowDegRate = 0 # 0= No , other wise , yes
        all_protein_all_batches = all_protein_all_batches.replace(np.inf, np.nan)
        all_peptide_all_batches = all_peptide_all_batches.replace(np.inf, np.nan)
        print('proteinoutlier removal is started')
        all_protein_all_batches = outlier_degPattern(all_protein_all_batches,FAD, WT, delete_3data_notfollowDegRate)
        print('proteinoutlier removal is complete')
    all_peptide_all_batches = all_peptide_all_batches[all_peptide_all_batches[['WT_day4', 'WT_day8', 'WT_day16', 'WT_day32','FAD_day4', 'FAD_day8', 'FAD_day16', 'FAD_day32']].count(axis=1) > 0]       

    if 'normalized_status' in all_protein_all_batches.columns:
        
        all_Prot_normalized     = (all_protein_all_batches['normalized_status'] == 'yes').count()
        all_Prot_not_normalized = len(all_protein_all_batches) - (all_protein_all_batches['normalized_status'] == 'yes').count()
        print("\n  all_Proteins quantified atleast in one batch = %d" %(len(all_protein_all_batches)))
        print("\n  all_proteins normalized with absolute protein intesnity =  %d" %(all_Prot_normalized) )
        print("\n  all_proteins NOT-normalized with absolute protein intesnity =  %d\n" %(all_Prot_not_normalized) )
                
        outfile = open(os.path.join(final_directory, r"all_protein_turnover_final.txt"),"w") 
        outfile.write("\n  Proteins quantified atleast in one batch = %d \n" %len(all_protein_all_batches))
        outfile.write("\n  Proteins normalised with absolute protein intesnity =  %d \n" %all_Prot_normalized )
        outfile.write("\n  Proteins NOT-normalised with absolute protein intesnity =  %d \n" %all_Prot_not_normalized )
        all_protein_all_batches.index.name = "Protein Accession"
        all_protein_all_batches.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
        outfile.close()
    else:
        print("\n  all_Proteins quantified atleast in one batch = %d \n" %(len(all_protein_all_batches)))
        outfile = open(os.path.join(final_directory, r"all_protein_turnover_final.txt"),"w") 
        outfile.write("\n  Proteins quantified atleast in one batch = %d \n" %len(all_protein_all_batches))
        all_protein_all_batches.index.name = "Protein Accession"
        all_protein_all_batches.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
        outfile.close()

    outfile = open(os.path.join(final_directory, r"peptide_turnover_final.txt"),"w") 
    all_peptide_all_batches.index.name = "Peptide"
    all_peptide_all_batches.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')#,index=None
    outfile.close()
  
    
    uni_protein_all_batches['b1_GN'] = uni_protein_all_batches.apply(lambda row: row['b2_GN'] if pd.isnull(row['b1_GN']) else row['b1_GN'],axis=1)
    uni_protein_all_batches['b1_Protein Description'] = uni_protein_all_batches.apply(lambda row: row['b2_Protein Description'] if pd.isnull(row['b1_Protein Description']) else row['b1_Protein Description'],axis=1)
    uni_protein_all_batches['b1_Protein Group#'] = uni_protein_all_batches.apply(lambda row: row['b2_Protein Group#'] if pd.isnull(row['b1_Protein Group#']) else row['b1_Protein Group#'],axis=1)
    if 'b1_normalized_status' in uni_protein_all_batches.columns:
        uni_protein_all_batches['b1_normalized_status'] = uni_protein_all_batches.apply(lambda row: row['b2_normalized_status'] if pd.isnull(row['b1_normalized_status']) else row['b1_normalized_status'],axis=1)
        uni_protein_all_batches = uni_protein_all_batches.drop(['b2_GN','b2_Protein Description','b2_normalized_status', 'b2_Protein Group#'], axis=1)
        uni_protein_all_batches.rename(columns={'b1_GN': 'GN','b1_Protein Description':'Protein Description', 'b1_normalized_status':'normalized_status', 'b1_Protein Group#': 'Protein Group'},inplace=True)
    else:
        uni_protein_all_batches = uni_protein_all_batches.drop(['b2_GN','b2_Protein Description','b2_Protein Group#'], axis=1)
        uni_protein_all_batches.rename(columns={'b1_GN': 'GN','b1_Protein Description':'Protein Description',  'b1_Protein Group#': 'Protein Group'},inplace=True)
                                    
    # Print Peptide data
    uni_protein_all_batches['WT_day0'] = (uni_protein_all_batches[[WT1[0], WT2[0], WT3[0]]]).mean(axis=1)
    uni_protein_all_batches['WT_day4'] = (uni_protein_all_batches[[WT1[1], WT2[1], WT3[1]]]).mean(axis=1)
    uni_protein_all_batches['WT_day8'] = (uni_protein_all_batches[[WT1[2], WT2[2], WT3[2]]]).mean(axis=1)
    uni_protein_all_batches['WT_day16'] = (uni_protein_all_batches[[WT1[3], WT2[3], WT3[3]]]).mean(axis=1)
    uni_protein_all_batches['WT_day32'] = (uni_protein_all_batches[[WT1[4], WT2[4], WT3[4]]]).mean(axis=1)
    
    uni_protein_all_batches['FAD_day0'] = (uni_protein_all_batches[[FAD1[0], FAD2[0], FAD3[0]]]).mean(axis=1)
    uni_protein_all_batches['FAD_day4'] = (uni_protein_all_batches[[FAD1[1], FAD2[1], FAD3[1]]]).mean(axis=1)
    uni_protein_all_batches['FAD_day8'] = (uni_protein_all_batches[[FAD1[2], FAD2[2], FAD3[2]]]).mean(axis=1)
    uni_protein_all_batches['FAD_day16'] = (uni_protein_all_batches[[FAD1[3], FAD2[3], FAD3[3]]]).mean(axis=1)
    uni_protein_all_batches['FAD_day32'] = (uni_protein_all_batches[[FAD1[4], FAD2[4], FAD3[4]]]).mean(axis=1)
    
    outfile = open(os.path.join(final_directory, r"uni_protein_turnover_final.txt"),"w") 
    uni_protein_all_batches.index.name = "Peptide"
    uni_protein_all_batches.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')#,index=None
    outfile.close()
    
    FAD = ['FAD_day0', 'FAD_day4', 'FAD_day8', 'FAD_day16', 'FAD_day32']
    WT = ['WT_day0', 'WT_day4', 'WT_day8', 'WT_day16', 'WT_day32']
    if params["free_Lys_outlier"] == "1": # Remove data points (at Peptide level) that are faster  than Lys dyamics
        for i in range(1, len(free_Lys_ratio)):
            uni_protein_all_batches[uni_protein_all_batches[[FAD[i], WT[i]]] <= free_Lys_ratio[i]] = np.nan
    if params["free_Lys_outlier"] == "2": # Remove data points (at Peptide level) that are faster  than Lys dyamics
        for i in range(1, len(free_Lys_ratio)):
            for j in range (0, len(uni_protein_all_batches)):
                if uni_protein_all_batches.iloc[j,   uni_protein_all_batches.columns.get_loc(FAD[i])] <= free_Lys_ratio[i]:
                    uni_protein_all_batches.iloc[j,  uni_protein_all_batches.columns.get_loc(FAD[i])]= free_Lys_ratio[i]
                if uni_protein_all_batches.iloc[j,   uni_protein_all_batches.columns.get_loc(WT[i])] <= free_Lys_ratio[i]:
                    uni_protein_all_batches.iloc[j,  uni_protein_all_batches.columns.get_loc(WT[i])]= free_Lys_ratio[i]    
    if params["degradation_pattern_outlier"] == "1": # Remove data points (at Peptide level) that do not follow degredation pattern
        delete_3data_notfollowDegRate = 0 # 0= No , other wise , yes
        uni_protein_all_batches = uni_protein_all_batches.replace(np.inf, np.nan)
        uni_protein_all_batches = outlier_degPattern(uni_protein_all_batches,FAD, WT, delete_3data_notfollowDegRate)
    uni_protein_all_batches = uni_protein_all_batches[uni_protein_all_batches[['WT_day4', 'WT_day8', 'WT_day16', 'WT_day32','FAD_day4', 'FAD_day8', 'FAD_day16', 'FAD_day32']].count(axis=1) > 0]       
    if 'normalized_status' in uni_protein_all_batches.columns:
        uni_Prot_normalized     = (uni_protein_all_batches['normalized_status'] == 'yes').count()
        uni_Prot_not_normalized = len(uni_protein_all_batches) - (uni_protein_all_batches['normalized_status'] == 'yes').count()
        print("\n  uni_Proteins quantified atleast in one batch = %d" %(len(uni_protein_all_batches)))
        print("\n  uni_proteins normalized with absolute protein intesnity =  %d" %(uni_Prot_normalized) )
        print("\n  uni_proteins NOT-normalized with absolute protein intesnity =  %d\n" %(uni_Prot_not_normalized) )
                    
        outfile = open(os.path.join(final_directory, r"uni_protein_turnover_final.txt"),"w") 
        outfile.write("\n  Proteins quantified atleast in one batch = %d \n" %len(uni_protein_all_batches))
        outfile.write("\n  Proteins normalised with absolute protein intesnity =  %d \n" %uni_Prot_normalized )
        outfile.write("\n  Proteins NOT-normalised with absolute protein intesnity =  %d\n" %uni_Prot_not_normalized )
        uni_protein_all_batches.index.name = "Protein Accession"
        uni_protein_all_batches.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
        outfile.close()
    else:
        print("\n  uni_Proteins quantified atleast in one batch = %d" %(len(uni_protein_all_batches)))
        outfile = open(os.path.join(final_directory, r"uni_protein_turnover_final.txt"),"w") 
        outfile.write("Proteins quantified atleast in one batch = %d \n" %len(uni_protein_all_batches))
        uni_protein_all_batches.index.name = "Protein Accession"
        uni_protein_all_batches.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
        outfile.close()

  
print("\n\n ********** Combined the two batches and the output is saved *************** \n\n" )



     