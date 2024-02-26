# -*- coding: utf-8 -*-
"""
SILAC-TMT data processing pipeline and the free lysine estimation pipeline
Last modified by Abhijit Dasgupta on February 06, 2024
@author: SChepyal and ADasgupta
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd, re, numpy as np, os, sys, pickle 
from scipy.optimize import curve_fit
from pyteomics import ms2, mzxml
from scipy.stats import t
import numpy.ma as ma
from collections import OrderedDict, Counter
from isotopeCalculation import *
pd.options.mode.chained_assignment = None
class OrderedCounter(Counter, OrderedDict):
	pass

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('logFile.log', 'a'))
print = logger.info

 #########################
 # Small module functions
 ########################
 
 
class progressBar:
    def __init__(self, total):
        self.total = total
        self.barLength = 20
        self.count = 0
        self.progress = 0
        self.block = 0
        self.status = ""

    def increment(self):
        self.count += 1
        self.progress = self.count / self.total
        self.block = int(round(self.barLength * self.progress))
        if self.progress == 1:
            self.status = "Done...\r\n"
        else:
            self.status = ""
        text = "\r  Progress: [{0}] {1}% {2}".format("#" * self.block + "-" * (self.barLength - self.block),
                                                     int(self.progress * 100), self.status)
        sys.stdout.write(text)
        sys.stdout.flush()
        
def getParams(paramFile):
    parameters = dict()
    with open(paramFile, 'r') as file:
        for line in file:
            #print("line1 = %s" %line)
            if re.search(r'^#', line) or re.search(r'^\s', line):
                continue
            line = re.sub(r'#.*', '', line)  # Remove comments (start from '#')
            line = re.sub(r'\s*', '', line)  # Remove all whitespaces

            # Exception for "feature_files" parameter
            if "feature_files" in parameters and line.endswith("feature"):
                parameters["feature_files"].append(line)
            else:
                key = line.split('=')[0]
                val = line.split('=')[1]
                if key == "feature_files":
                    parameters[key] = [val]
                else:
                    parameters[key] = val
    return parameters

def extractReporters(files, df, params, **kwargs):
    # Input arguments
    # files: mzXML or ms2 files to be quantified
    # df: dataframe of ID.txt file
    # params: parameters
    if "sig126" in kwargs:
        print("\n  Refined extraction of TMT reporter ion peaks")
    else:
        print("\n  Extraction of TMT reporter ion peaks")

    dictQuan = {}
    for file in files:
        print("    Working on {}".format(os.path.basename(file)))
        ext = os.path.splitext(file)[-1]
        if ext == ".mzXML":
            reader = mzxml.MzXML(file)  # mzXML file reader
        elif ext == ".ms2":
            reader = ms2.IndexedMS2(file)  # MS2 file reader
        else:
            sys.exit(" Currently, either .mzXML or .ms2 file is supported")

        # Extraction of TMT reporter ions in each fraction
        scans = list(df['scan'][df['frac'] == file].unique())
        progress = progressBar(len(scans))
        for scan in scans:
            progress.increment()
            spec = reader[str(scan)]
            res = getReporterIntensity(spec, params, **kwargs)  # Array of reporter m/z and intensity values
            key = file + "_" + str(scan)
            dictQuan[key] = res

    # Create a dataframe of quantification data
    reporters = params["tmt_reporters_used"].split(";")
    colNames = [re.sub("sig", "mz", i) for i in reporters] + reporters
    res = pd.DataFrame.from_dict(dictQuan, orient='index', columns=colNames)

    # Summary of quantified TMT reporter ions
    reporterSummary = getReporterSummary(res, reporters)
    nTot = len(res)
    for reporter in reporters:
        n = reporterSummary[reporter]["nPSMs"]
        print("    %s\t%d (%.2f%%) matched" % (reporter, n, n / nTot * 100))

    return res, reporterSummary

def getReporterIntensity(spec, params, **kwargs):
    tol = 10
    reporterNames = params["tmt_reporters_used"].split(";")
    mzArray = []
    intensityArray = []

    for reporter in reporterNames:
        if reporter in kwargs:
            mz = getReporterMz(reporter) * (1 + kwargs[reporter]["meanMzShift"] / 1e6)
            tol = kwargs[reporter]["sdMzShift"] * np.float(params['tmt_peak_extraction_second_sd'])
        else:
            mz = getReporterMz(reporter)

        lL = mz - mz * tol / 1e6
        uL = mz + mz * tol / 1e6
        ind = np.where((spec["m/z array"] >= lL) & (spec["m/z array"] <= uL))[0]
        if len(ind) == 0:
            mz = 0
        elif len(ind) == 1:
            ind = ind[0]
            mz = spec["m/z array"][ind]
        elif len(ind) > 1:
            if params['tmt_peak_extraction_method'] == '2':
                ind2 = np.argmin(abs(mz - spec["m/z array"][ind]))
                ind = ind[ind2]
                mz = spec["m/z array"][ind]
            else:
                ind2 = np.argmax(spec["intensity array"][ind])
                ind = ind[ind2]
                mz = spec["m/z array"][ind]
        if lL <= mz < uL:
            intensity = spec["intensity array"][ind]
        else:
            intensity = 0
        mzArray.append(mz)
        intensityArray.append(intensity)

    outArray = mzArray + intensityArray
    return outArray


def getReporterMz(name):
    if name == "sig126":
        return 126.127726
    elif name == "sig127" or name == "sig127N":
        return 127.124761
    elif name == "sig127C":
        return 127.131081
    elif name == "sig128N":
        return 128.128116
    elif name == "sig128" or name == "sig128C":
        return 128.134436
    elif name == "sig129" or name == "sig129N":
        return 129.131471
    elif name == "sig129C":
        return 129.137790
    elif name == "sig130N":
        return 130.134825
    elif name == "sig130" or name == "sig130C":
        return 130.141145
    elif name == "sig131" or name == "sig131N":
        return 131.138180
    elif name == "sig131C":
        return 131.144500
    elif name == "sig132N":
        return 132.141535
    elif name == "sig132C":
        return 132.147855
    elif name == "sig133N":
        return 133.144890
    elif name == "sig133C":
        return 133.151210
    elif name == "sig134N":
        return 134.148245


def getReporterSummary(df, reporters):
    print("  Summary of quantified TMT reporter ions")
    res = {}
    for reporter in reporters:
        res[reporter] = {}
        reporterMz = getReporterMz(reporter)
        measuredMz = df[reporter.replace("sig", "mz")]
        measuredMz = measuredMz[measuredMz > 0]
        n = len(measuredMz)
        meanMzShift = ((measuredMz - reporterMz) / reporterMz * 1e6).mean()
        sdMzShift = ((measuredMz - reporterMz) / reporterMz * 1e6).std(ddof=1)
        res[reporter]['nPSMs'] = n
        res[reporter]['meanMzShift'] = meanMzShift
        res[reporter]['sdMzShift'] = sdMzShift

    return res


def getSubset(df, params):
    # Get a subset of a dataframe to calculate loading-bias information
    # 1. Filter out PSMs based on the intensity level
    reporters = params["tmt_reporters_used"].split(";")
    noiseLevel = 1000
    snRatio = float(params["SNratio_for_correction"])
    subDf = df[reporters][(df[reporters] > noiseLevel * snRatio).prod(axis=1).astype(bool)]  # Zero-intensity PSMs are excluded

    # 2. Filter out highly variant PSMs in each column (reporter)
    psmMean = subDf.mean(axis=1)
    subDf = np.log2(subDf.divide(psmMean, axis=0))
    pctTrimmed = float(params["percentage_trimmed"])
    n = 0
    for reporter in reporters:
        if n == 0:
            ind = ((subDf[reporter] > subDf[reporter].quantile(pctTrimmed / 200)) &
                   (subDf[reporter] < subDf[reporter].quantile(1 - pctTrimmed / 200)))
        else:
            ind = ind & ((subDf[reporter] > subDf[reporter].quantile(pctTrimmed / 200)) &
                         (subDf[reporter] < subDf[reporter].quantile(1 - pctTrimmed / 200)))
        n += 1
    subDf = subDf.loc[ind]
    return subDf

def matchMs2ToMs1(scans, reader):
    output = {} # Dictionary whose key = MS2 scan number (string), value = corresponding MS3 scan number (string)
    for scan in scans:
        i = np.int(scan) - 1
        spec = reader[str(i)]
        while spec["msLevel"] > 0:
            if spec["msLevel"] == 1:
                ms1Num = spec["num"]
                output[str(scan)] = ms1Num
                break
            i -= 1
            spec = reader[str(i)]
    return output

def getMS1ReporterIntensity(spec, params,iso_distr, charge, pep_identified):
    mzArray = []
    intensityArray = []
    tol = np.float(params['mass_tolerance_ms1'])
    isoDiffC = 1.00335
    mz_ = iso_distr[0] - isoDiffC / np.int(charge);
    iso_distr = np.insert(iso_distr, 0, mz_, axis=0)
    for ix in iso_distr: 
        mz = ix
        lL = mz - mz * tol / 1e6
        uL = mz + mz * tol / 1e6
        ind = np.where((spec["m/z array"] >= lL) & (spec["m/z array"] <= uL))[0]
        if len(ind) == 0:
            reporterMz = 0
        elif len(ind) == 1:
            ind = ind[0]
            reporterMz = spec["m/z array"][ind]
        elif len(ind) > 1:
            if params['ms1_peak_extraction_method'] == '2':
                ind2 = np.argmin(abs(mz - spec["m/z array"][ind]))
                ind = ind[ind2]
                reporterMz = spec["m/z array"][ind]
            else:
                ind2 = np.argmax(spec["intensity array"][ind])
                ind = ind[ind2]
                reporterMz = spec["m/z array"][ind]
        if lL <= reporterMz < uL:
            reporterIntensity = spec["intensity array"][ind]
        else:
            reporterIntensity = 0
        mzArray.append(reporterMz)
        intensityArray.append(reporterIntensity)
    MS1_peak_intensity = intensityArray[1]
    MS1_peak_mz = mzArray[1]
    ## When there are enough matches between theoretical and observed peaks, those matches are further investigated 
    if (intensityArray[0] > MS1_peak_intensity):
        MS1_peak_intensity = 0
        MS1_peak_mz = 0
            
    return MS1_peak_intensity, MS1_peak_mz

def getMS1ReporterIntensity_lys(spec, params,iso_distr, charge, pep_identified):
    mzArray = []
    intensityArray = []
    tol = np.float(params['mass_tolerance_ms1'])
    isoDiffC = 1.00335
    mz_ = iso_distr[0] - isoDiffC / np.int(charge);
    iso_distr = np.insert(iso_distr, 0, mz_, axis=0)
    for ix in iso_distr: 
        mz = ix
        lL = mz - mz * tol / 1e6
        uL = mz + mz * tol / 1e6
        ind = np.where((spec["m/z array"] >= lL) & (spec["m/z array"] <= uL))[0]
        if len(ind) == 0:
            reporterMz = 0
        elif len(ind) == 1:
            ind = ind[0]
            reporterMz = spec["m/z array"][ind]
        elif len(ind) > 1:
            if params['ms1_peak_extraction_method'] == '2':
                ind2 = np.argmin(abs(mz - spec["m/z array"][ind]))
                ind = ind[ind2]
                reporterMz = spec["m/z array"][ind]
            else:
                ind2 = np.argmax(spec["intensity array"][ind])
                ind = ind[ind2]
                reporterMz = spec["m/z array"][ind]
        if lL <= reporterMz < uL:
            reporterIntensity = spec["intensity array"][ind]
        else:
            reporterIntensity = 0
        mzArray.append(reporterMz)
        intensityArray.append(reporterIntensity)
    MS1_peak_intensity = intensityArray[1]+intensityArray[2]
    MS1_peak_mz = mzArray[1]
    ## When there are enough matches between theoretical and observed peaks, those matches are further investigated 
    if (intensityArray[0] > MS1_peak_intensity):
        MS1_peak_intensity = 0
        MS1_peak_mz = 0
            
    return MS1_peak_intensity, MS1_peak_mz


def get_isope_distri(df, params, **kwargs):
    
    #preComputedIsotopes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isotopeMassIntensity.pkl")
    preComputedIsotopes = os.path.join(os.path.dirname(os.path.abspath("isotopeMassIntensity.pkl")),"isotopeMassIntensity.pkl" )

    # Open the default elementary dictionary
    with open(preComputedIsotopes, 'rb') as f:
        iso_mass_inten_dict = pickle.load(f)

    if 'PTM_Phosphorylation' in params:
        if any(params['PTM_Phosphorylation']):
            std_aa_comp.update({params['PTM_phosphorylation']: {'H': 1, 'P': 1, 'O': 3}})
        else:
            std_aa_comp.update({'#': {'H': 1, 'P': 1, 'O': 3}})

    if 'PTM_mono_oxidation' in params:
        std_aa_comp.update({params['PTM_mono_oxidation']: {'O': 1}})
    
    std_aa_comp.update({'&': {'C': -6, 'x': 6}}) # upadte elemnt dictionary with heavy lysin param
    
    # Update the elementary dictionary with user defined tracer elements and their natural abundance
    elemInfo_dict = {}
    if ('Tracer_1' in params):
        if (params['Tracer_1'] == '13C'):
            elemInfo_dict.update({"x": {12: 1 - float(params['Tracer_1_purity']),
                                        13.00335483521: float(params['Tracer_1_purity'])}})
        elif (params['Tracer_1'] == '15N'):
            elemInfo_dict.update({"y": {14.00307400446: 1 - float(params['Tracer_1_purity']),
                                        15.0001088989: float(params['Tracer_1_purity'])}})
        else:
            print("\n Information for parameter \"Tracer_1\" is not properly defined\n ")

    if ('Tracer_2' in params):
        if (params['Tracer_2'] == '13C'):
            elemInfo_dict.update({"x": {12: 1 - float(params['Tracer_2_purity']),
                                        13.00335483521: float(params['Tracer_2_purity'])}})
        elif (params['Tracer_2'] == '15N'):
            elemInfo_dict.update({"y": {14.00307400446: 1 - float(params['Tracer_2_purity']),
                                        15.0001088989: float(params['Tracer_2_purity'])}})
        else:
            print("\n Information for parameter \"Tracer_2\" is not properly defined\n ")
    iso_mass_inten_dict = isotope_distribution_indElement(elemInfo_dict, iso_mass_inten_dict, inten_threshold_trim=1e-10)        
    dictPepIsoDist = {}
    print("\n Calculating the isotopic distribution for the peptides\n ")
    df["Charge"] = df["Outfile"].apply(lambda x: os.path.basename(x).split(".")[-2])
    df2 = df[['Peptide', 'Charge']]
    df2["Peptide"] = df2["Peptide"].str.replace('&', '')
    df2['Peptide'] = df2['Peptide'].apply(lambda x: x.split(".")[1])
    df2["key"] = df2["Peptide"] + "_" + df2["Charge"]
    df2 = df2.drop_duplicates(["key"])
    progress = progressBar(len(df2))

    for pep in df2['key'].unique():
        progress.increment()
        peptide_, charge = pep.split("_")
        charge = np.int(charge)
        if peptide_.count('K') == 1:
            chemical_com_ = pepSeq_to_chemComp(peptide_, charge, std_aa_comp, 'TMTPro')# TMT_Plex == 'TMTPro' or 'TMT10' or 'SILAC'
            chemical_com_ = {k: v for k, v in chemical_com_.items() if v != 0}
            iso_distr = iso_distri(iso_mass_inten_dict, chemical_com_, charge, float(params['isotope_cutoff']), float(params['mass_tolerance']), float(params['method_merging_isotopic_peaks']), is_pep=1)
            iso_distr = iso_distr.sort_values(['isotope_inten'], ascending=[False]).reset_index(drop = True)
            iso_distr = iso_distr.loc[0:2].reset_index(drop = True)
            
            dictPepIsoDist[peptide_+'_'+str(charge)] = peptide_, charge, iso_distr.isotope_mass.values, iso_distr.isotope_inten.values 
             
            peptide_ = peptide_.replace('K', 'K&')
            chemical_com_ = pepSeq_to_chemComp(peptide_, charge, std_aa_comp, 'TMTPro')# TMT_Plex == 'TMTPro' or 'TMT10' or 'SILAC'
            chemical_com_ = {k: v for k, v in chemical_com_.items() if v != 0}
            iso_distr = iso_distri(iso_mass_inten_dict, chemical_com_, charge, float(params['isotope_cutoff']), float(params['mass_tolerance']), float(params['method_merging_isotopic_peaks']), is_pep=1)
            iso_distr = iso_distr.sort_values(['isotope_inten'], ascending=[False]).reset_index(drop = True)
            iso_distr = iso_distr.loc[0:2].reset_index(drop = True)
                                    
            dictPepIsoDist[peptide_+'_'+str(charge)] = peptide_, charge, iso_distr.isotope_mass.values, iso_distr.isotope_inten.values            
    # Create a dataframe of quantification data
    pepIsoDist = pd.DataFrame.from_dict(dictPepIsoDist, orient='index', columns=['Peptide', 'Charge', 'isotope_distribution', 'isotope_inten'])

    return pepIsoDist

def get_isope_distri_lys(df, params, **kwargs):
    #preComputedIsotopes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isotopeMassIntensity.pkl")
    preComputedIsotopes = os.path.join(os.path.dirname(os.path.abspath("isotopeMassIntensity.pkl")),"isotopeMassIntensity.pkl" )

    # Open the default elementary dictionary
    with open(preComputedIsotopes, 'rb') as f:
        iso_mass_inten_dict = pickle.load(f)

    if 'PTM_Phosphorylation' in params:
        if any(params['PTM_Phosphorylation']):
            std_aa_comp.update({params['PTM_phosphorylation']: {'H': 1, 'P': 1, 'O': 3}})
        else:
            std_aa_comp.update({'#': {'H': 1, 'P': 1, 'O': 3}})

    if 'PTM_mono_oxidation' in params:
        std_aa_comp.update({params['PTM_mono_oxidation']: {'O': 1}})
    
    std_aa_comp.update({'&': {'C': -6, 'x': 6}}) # upadte elemnt dictionary with heavy lysin param
    
    # Update the elementary dictionary with user defined tracer elements and their natural abundance
    elemInfo_dict = {}
    if ('Tracer_1' in params):
        if (params['Tracer_1'] == '13C'):
            elemInfo_dict.update({"x": {12: 1 - float(params['Tracer_1_purity']),
                                        13.00335483521: float(params['Tracer_1_purity'])}})
        elif (params['Tracer_1'] == '15N'):
            elemInfo_dict.update({"y": {14.00307400446: 1 - float(params['Tracer_1_purity']),
                                        15.0001088989: float(params['Tracer_1_purity'])}})
        else:
            print("\n Information for parameter \"Tracer_1\" is not properly defined\n ")

    if ('Tracer_2' in params):
        if (params['Tracer_2'] == '13C'):
            elemInfo_dict.update({"x": {12: 1 - float(params['Tracer_2_purity']),
                                        13.00335483521: float(params['Tracer_2_purity'])}})
        elif (params['Tracer_2'] == '15N'):
            elemInfo_dict.update({"y": {14.00307400446: 1 - float(params['Tracer_2_purity']),
                                        15.0001088989: float(params['Tracer_2_purity'])}})
        else:
            print("\n Information for parameter \"Tracer_2\" is not properly defined\n ")
    iso_mass_inten_dict = isotope_distribution_indElement(elemInfo_dict, iso_mass_inten_dict, inten_threshold_trim=1e-10)        
    dictPepIsoDist = {}
    print("\n Calculating the isotopic distribution for the peptides\n ")
    df["Charge"] = df["Outfile"].apply(lambda x: os.path.basename(x).split(".")[-2])
    df2 = df[['Peptide', 'Charge']]
    df2["Peptide"] = df2["Peptide"].str.replace('&', '')
    df2['Peptide'] = df2['Peptide'].apply(lambda x: x.split(".")[1])
    df2["key"] = df2["Peptide"] + "_" + df2["Charge"]
    df2 = df2.drop_duplicates(["key"])
    progress = progressBar(len(df2))

    for pep in df2['key'].unique():
        progress.increment()
        peptide_, charge = pep.split("_")
        charge = np.int(charge)
        if peptide_.count('K') > 1:
            chemical_com_ = pepSeq_to_chemComp(peptide_, charge, std_aa_comp, 'TMTPro')# TMT_Plex == 'TMTPro' or 'TMT10' or 'SILAC'
            chemical_com_ = {k: v for k, v in chemical_com_.items() if v != 0}
            iso_distr = iso_distri(iso_mass_inten_dict, chemical_com_, charge, float(params['isotope_cutoff']), float(params['mass_tolerance']), float(params['method_merging_isotopic_peaks']), is_pep=1)
            iso_distr = iso_distr.sort_values(['isotope_inten'], ascending=[False]).reset_index(drop = True)
            iso_distr = iso_distr.loc[0:2].reset_index(drop = True)
            dictPepIsoDist[peptide_+'_'+str(charge)] = peptide_, charge, iso_distr.isotope_mass.values, iso_distr.isotope_inten.values 
            
            peptide_1 = re.sub(r'^(.*?(K.*?){0})K', r'\1K&', peptide_)
            chemical_com_ = pepSeq_to_chemComp(peptide_1, charge, std_aa_comp, 'TMTPro')# TMT_Plex == 'TMTPro' or 'TMT10' or 'SILAC'
            chemical_com_ = {k: v for k, v in chemical_com_.items() if v != 0}
            iso_distr = iso_distri(iso_mass_inten_dict, chemical_com_, charge, float(params['isotope_cutoff']), float(params['mass_tolerance']), float(params['method_merging_isotopic_peaks']), is_pep=1)
            iso_distr = iso_distr.sort_values(['isotope_inten'], ascending=[False]).reset_index(drop = True)
            iso_distr = iso_distr.loc[0:2].reset_index(drop = True)
            dictPepIsoDist[peptide_1+'_'+str(charge)] = peptide_1, charge, iso_distr.isotope_mass.values, iso_distr.isotope_inten.values            
            
            peptide_2 = re.sub(r'^(.*?(K.*?){1})K', r'\1K&', peptide_)
            chemical_com_ = pepSeq_to_chemComp(peptide_2, charge, std_aa_comp, 'TMTPro')# TMT_Plex == 'TMTPro' or 'TMT10' or 'SILAC'
            chemical_com_ = {k: v for k, v in chemical_com_.items() if v != 0}
            iso_distr = iso_distri(iso_mass_inten_dict, chemical_com_, charge, float(params['isotope_cutoff']), float(params['mass_tolerance']), float(params['method_merging_isotopic_peaks']), is_pep=1)
            iso_distr = iso_distr.sort_values(['isotope_inten'], ascending=[False]).reset_index(drop = True)
            iso_distr = iso_distr.loc[0:2].reset_index(drop = True)
            dictPepIsoDist[peptide_2+'_'+str(charge)] = peptide_2, charge, iso_distr.isotope_mass.values, iso_distr.isotope_inten.values            

            peptide_ = peptide_.replace('K', 'K&')
            chemical_com_ = pepSeq_to_chemComp(peptide_, charge, std_aa_comp, 'TMTPro')# TMT_Plex == 'TMTPro' or 'TMT10' or 'SILAC'
            chemical_com_ = {k: v for k, v in chemical_com_.items() if v != 0}
            iso_distr = iso_distri(iso_mass_inten_dict, chemical_com_, charge, float(params['isotope_cutoff']), float(params['mass_tolerance']), float(params['method_merging_isotopic_peaks']), is_pep=1)
            iso_distr = iso_distr.sort_values(['isotope_inten'], ascending=[False]).reset_index(drop = True)
            iso_distr = iso_distr.loc[0:2].reset_index(drop = True)
            dictPepIsoDist[peptide_+'_'+str(charge)] = peptide_, charge, iso_distr.isotope_mass.values, iso_distr.isotope_inten.values            
    # Create a dataframe of quantification data
    pepIsoDist = pd.DataFrame.from_dict(dictPepIsoDist, orient='index', columns=['Peptide', 'Charge', 'isotope_distribution', 'isotope_inten'])

    return pepIsoDist

def extract_MS1_intesnity(file, df_, pepIsoDist, params, **kwargs):
    dictQuan = {}
    ext = os.path.splitext(file)[-1]
    if ext == ".mzXML":
        reader = mzxml.MzXML(file)  # mzXML file reader
    elif ext == ".ms2":
        reader = ms2.IndexedMS2(file)  # MS2 file reader
    else:
        sys.exit(" Currently, either .mzXML or .ms2 file is supported")
    scans = list(df['scan'].unique())
    ms2ToMs1 = matchMs2ToMs1(scans, reader)
    for scan in scans:
        peptide = list(df['Peptide'][df['scan'] == scan].unique())[0]
        peptide_ = peptide.split(".")[1]
        if peptide_.count('K') == 1:
            charge = ((list(df['Outfile'][df['scan'] == scan].unique())[0]).split('.')[-2])
            if peptide_.count('&') == 1:
                pep_identified = 1
                iso_distr = pepIsoDist.loc[peptide_+'_'+charge, 'isotope_distribution']
                if isinstance(iso_distr, str):
                    iso_distr = np.fromstring(iso_distr.strip(']['), dtype=float, sep=' ')
                ms1_scan = ms2ToMs1[scan]
                spec = reader[str(ms1_scan)]
                heavy_MS1_peak_mz_theo = iso_distr[0]
                heavyInt, heavy_MS1_peak_mz = getMS1ReporterIntensity(spec, params,iso_distr, np.int(charge), pep_identified)
                    
                peptide_ = peptide_.replace('&', '')
                pep_identified = 0
                iso_distr = pepIsoDist.loc[peptide_+'_'+charge, 'isotope_distribution']
                if isinstance(iso_distr, str):
                    iso_distr = np.fromstring(iso_distr.strip(']['), dtype=float, sep=' ')
                light_MS1_peak_mz_theo = iso_distr[0]
                lightInt, light_MS1_peak_mz = getMS1ReporterIntensity(spec, params,iso_distr, np.int(charge), pep_identified)
            else:
                pep_identified = 1
                iso_distr = pepIsoDist.loc[peptide_+'_'+charge, 'isotope_distribution']
                if isinstance(iso_distr, str):
                    iso_distr = np.fromstring(iso_distr.strip(']['), dtype=float, sep=' ')                    
                light_MS1_peak_mz_theo = iso_distr[0]
                ms1_scan = ms2ToMs1[scan]
                spec = reader[str(ms1_scan)]
                    
                lightInt, light_MS1_peak_mz = getMS1ReporterIntensity(spec, params,iso_distr, np.int(charge), pep_identified)
                    
                peptide_ = peptide_.replace('K', 'K&')
                pep_identified = 0
                iso_distr = pepIsoDist.loc[peptide_+'_'+charge, 'isotope_distribution']
                if isinstance(iso_distr, str):
                    iso_distr = np.fromstring(iso_distr.strip(']['), dtype=float, sep=' ')
                heavy_MS1_peak_mz_theo = iso_distr[0]
                heavyInt, heavy_MS1_peak_mz = getMS1ReporterIntensity(spec, params,iso_distr, np.int(charge), pep_identified)
        key = file + "_" + str(scan)
        dictQuan[key] = [lightInt, heavyInt, light_MS1_peak_mz, heavy_MS1_peak_mz, light_MS1_peak_mz_theo, heavy_MS1_peak_mz_theo]
    # Create a dataframe of quantification data
    res = pd.DataFrame.from_dict(dictQuan, orient='index', columns=['light_intensity', 'heavy_intensity', 'light_mz_obs', 'heavy_mz_obs', 'light_mz_theo', 'heavy_mz_theo'])
    return res
def extract_MS1_intesnity_for_lys(file, df_, pepIsoDist, params, **kwargs):
    dictQuan = {}
    ext = os.path.splitext(file)[-1]
    if ext == ".mzXML":
        reader = mzxml.MzXML(file)  # mzXML file reader
    elif ext == ".ms2":
        reader = ms2.IndexedMS2(file)  # MS2 file reader
    else:
        sys.exit(" Currently, either .mzXML or .ms2 file is supported")
    scans = list(df['scan'].unique())
    ms2ToMs1 = matchMs2ToMs1(scans, reader)
    for scan in scans:
        peptide = list(df['Peptide'][df['scan'] == scan].unique())[0]
        peptide_ = peptide.split(".")[1]
        if peptide_.count('K') == 2:
            charge = ((list(df['Outfile'][df['scan'] == scan].unique())[0]).split('.')[-2])
            if peptide_.count('&') == 1:
                pep_identified = 1
                iso_distr = pepIsoDist.loc[peptide_+'_'+charge, 'isotope_distribution']
                if isinstance(iso_distr, str):
                    iso_distr = np.fromstring(iso_distr.strip(']['), dtype=float, sep=' ')
                ms1_scan = ms2ToMs1[scan]
                spec = reader[str(ms1_scan)]
                mixed_MS1_peak_mz_theo = iso_distr[0]
                mixed_MS1Int, mixed_MS1_peak_mz = getMS1ReporterIntensity_lys(spec, params,iso_distr, np.int(charge), pep_identified)
                    
                peptide_ = peptide_.replace('&', '')
                peptide_ = peptide_.replace('K', 'K&')
                pep_identified = 0
                iso_distr = pepIsoDist.loc[peptide_+'_'+charge, 'isotope_distribution']
                if isinstance(iso_distr, str):
                    iso_distr = np.fromstring(iso_distr.strip(']['), dtype=float, sep=' ')
                heavy_MS1_peak_mz_theo = iso_distr[0]
                heavy_MS1Int, heavy_MS1_peak_mz = getMS1ReporterIntensity_lys(spec, params,iso_distr, np.int(charge), pep_identified)
            elif peptide_.count('&') == 2:
                pep_identified = 1
                iso_distr = pepIsoDist.loc[peptide_+'_'+charge, 'isotope_distribution']
                if isinstance(iso_distr, str):
                    iso_distr = np.fromstring(iso_distr.strip(']['), dtype=float, sep=' ')
                ms1_scan = ms2ToMs1[scan]
                spec = reader[str(ms1_scan)]
                heavy_MS1_peak_mz_theo = iso_distr[0]
                heavy_MS1Int, heavy_MS1_peak_mz = getMS1ReporterIntensity_lys(spec, params,iso_distr, np.int(charge), pep_identified)
                    
                peptide_ = peptide_.replace('&', '')
                peptide_ = peptide_.replace('K', 'K&',1)
                pep_identified = 0
                iso_distr = pepIsoDist.loc[peptide_+'_'+charge, 'isotope_distribution']
                if isinstance(iso_distr, str):
                    iso_distr = np.fromstring(iso_distr.strip(']['), dtype=float, sep=' ')
                mixed_MS1_peak_mz_theo = iso_distr[0]
                mixed_MS1Int, mixed_MS1_peak_mz = getMS1ReporterIntensity_lys(spec, params,iso_distr, np.int(charge), pep_identified)
           
        key = file + "_" + str(scan)
        dictQuan[key] = [mixed_MS1Int, heavy_MS1Int, mixed_MS1_peak_mz, heavy_MS1_peak_mz, mixed_MS1_peak_mz_theo, heavy_MS1_peak_mz_theo]
    # Create a dataframe of quantification data
    res = pd.DataFrame.from_dict(dictQuan, orient='index', columns=['mixed_MS1_intensity','heavy_MS1_intensity', 'mixed_MS1_mz_obs', 'heavy_MS1_mz_obs', 'mixed_MS1_mz_theo', 'heavy_MS1_mz_theo',])
    return res
def correctImpurity(df, params):
    #This function is slighly modified (last line) from the original function used in jump-q
    if params['impurity_correction'] == "1":
        reporters = params["tmt_reporters_used"].split(";")
        dfImpurity = pd.read_table(params["impurity_matrix"], sep="\t", skiprows=1, header=None, index_col=0)
        dfImpurity = pd.DataFrame(np.linalg.pinv(dfImpurity.values), dfImpurity.columns, dfImpurity.index)
        dfCorrected = df[reporters].dot(dfImpurity.T)
        dfCorrected.columns = reporters
        #df[reporters] = pd.concat([df[reporters]/2, dfCorrected]).groupby(level=0).max()
        df[reporters] = dfCorrected
    return df

def normalization_Lys_PSM(df, df_2, params):
    ################################################
    # Normalization (i.e. loading-bias correction) #
    ################################################
    doNormalization = params["loading_bias_correction"]
    normalizationMethod = params["loading_bias_correction_method"]
    if doNormalization == "1":
        subDf = getSubset(df, params)
        # Calculate normalization factors for reporters
        print("  Normalization is being performed")
        if normalizationMethod == "1":  # Trimmed-mean
            normFactor = subDf.mean(axis=0) - np.mean(subDf.mean())
        elif normalizationMethod == "2":  # Trimmed-median
            normFactor = subDf.median(axis=0) - np.mean(subDf.median())

        # Normalize the non-Lys containing PSMs
        psmMean = df[reporters].mean(axis=1)
        df[reporters] = np.log2(df[reporters].divide(psmMean, axis=0).replace(0, np.nan))
        df[reporters] = df[reporters] - normFactor
        df[reporters] = 2 ** df[reporters]
        df[reporters] = df[reporters].multiply(psmMean, axis=0).replace(np.nan, 0)
        
        # Normalize the single Lys containg PSMs
        psmMean = df_2[reporters].mean(axis=1)
        df_2[reporters] = np.log2(df_2[reporters].divide(psmMean, axis=0).replace(0, np.nan))
        df_2[reporters] = df_2[reporters] - normFactor
        df_2[reporters] = 2 ** df_2[reporters]
        df_2[reporters] = df_2[reporters].multiply(psmMean, axis=0).replace(np.nan, 0)
    else:
        print("  Normalization is skipped according to the parameter")

    return df, df_2


def Qtest(data, left=True, right=True, alpha=0.05):
 
    q90 = [0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437,
           0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
           0.32, 0.313, 0.306, 0.3, 0.295, 0.29, 0.285, 0.281,
           0.277, 0.273, 0.269, 0.266, 0.263, 0.26
           ]
    Q90 = {n: q for n, q in zip(range(3, len(q90) + 1), q90)}
    q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
           0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
           0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
           0.308, 0.305, 0.301, 0.29
           ]
    Q95 = {n: q for n, q in zip(range(3, len(q95) + 1), q95)}
    q99 = [0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568,
           0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442,
           0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388,
           0.384, 0.38, 0.376, 0.372
           ]
    Q99 = {n: q for n, q in zip(range(3, len(q99) + 1), q99)}

    if isinstance(data, list):
        pass
    else:
        x = list(data)

    if alpha == 0.1:
        q_dict = Q90
    elif alpha == 0.05:
        q_dict = Q95
    elif alpha == 0.01:
        q_dict = Q99

    assert(left or right), 'At least one of the variables, `left` or `right`, must be True.'
    assert(len(data) >= 3), 'At least 3 data points are required'
    assert(len(data) <= max(q_dict.keys())), 'Sample size too large'

    sdata = sorted(data)
    Q_mindiff, Q_maxdiff = (0,0), (0,0)

    if left:
        Q_min = (sdata[1] - sdata[0])
        try:
            Q_min /= (sdata[-1] - sdata[0])
        except ZeroDivisionError:
            pass
        Q_mindiff = (Q_min - q_dict[len(data)], sdata[0])

    if right:
        Q_max = abs((sdata[-2] - sdata[-1]))
        try:
            Q_max /= abs((sdata[0] - sdata[-1]))
        except ZeroDivisionError:
            pass
        Q_maxdiff = (Q_max - q_dict[len(data)], sdata[-1])

    if not Q_mindiff[0] > 0 and not Q_maxdiff[0] > 0:
        outliers = []
    elif Q_mindiff[0] == Q_maxdiff[0]:
        outliers = [Q_mindiff[1], Q_maxdiff[1]]
    elif Q_mindiff[0] > Q_maxdiff[0]:
        outliers = [Q_mindiff[1]]
    else:
        outliers = [Q_maxdiff[1]]

    outlierInd = [i for i, v in enumerate(data) if v in outliers]

    return outlierInd

def outlierRemoval(df_, alpha, reporters):
    if isinstance(reporters, str):
        df = df_[reporters].to_frame()
    else:
        df = df_[reporters].copy()
    n = len(df)
    nOutliers = int(np.round(n * 0.3))
    indArray = []
    if nOutliers > n - 2:
        nOutliers = n - 2
    if nOutliers > 1:
        for i in range(df.shape[1]):
            ind = ESDtest(df.iloc[:, i], alpha, nOutliers)
            indArray.extend(ind)
    else:
        if n > 10:
            for i in range(df.shape[1]):
                ind = ESDtest(df.iloc[:, i], alpha, nOutliers)
                indArray.extend(ind)
        elif n > 2:
            for i in range(df.shape[1]):
                ind = Qtest(df.iloc[:, i], alpha, nOutliers)
                indArray.extend(ind)
    # PSMs including one or more outliers will not be considered for the subsequent quantification
    indArray_ = list(set(indArray))    # Indices of outliers across all reporters
    if len(indArray_) == n:
        c = OrderedCounter(indArray)
        keys = list(c)
        indArray = np.array(sorted(c, key=lambda x: (-c[x], keys.index(x)))).astype(int) 
        indArray = indArray[0:nOutliers]
    else:
        indArray = list(set(indArray))    
    df_.drop(df_.index[indArray], axis=0, inplace=True)
    return df_

def getLoadingBias(df, params):
    ###########################
    # Loading-bias evaluation #
    ###########################
    subDf   = getSubset(df, params)
    n       = len(subDf)
    sm      = 2 ** subDf.mean(axis=0)    # Sample-mean values
    msm     = np.mean(sm)    # Mean of sample-mean values
    avg     = sm / msm * 100
    sdVal   = subDf.std(axis=0, ddof = 1)
    sd      = ((2 ** sdVal - 1) + (1 - 2 ** (-sdVal))) / 2 * 100
    sem     = sd / np.sqrt(n)
    return avg, sd, sem, n


def light_perc(df,rep_temp):
    for cl in reversed(rep_temp):
        df[cl] = df[cl]/df[rep_temp[0]]
    return df

def func(x, a, b):
    return np.exp(-b * x)

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

def ESDtest(x, alpha, maxOLs):
    xm = np.ma.masked_invalid(x)
    n = len(xm)
    R, L, minds = [], [], []
    for i in range(maxOLs):
        xmean = xm.mean()
        xstd = xm.std(ddof=1)
        # Find maximum deviation
        rr = np.abs((xm - xmean) / xstd)
        minds.append(np.argmax(rr))
        R.append(rr[minds[-1]])
        p = 1.0 - alpha / (2.0 * (n - i))
        perPoint = t.ppf(p, n - i - 2)
        L.append((n - i-1) * perPoint / np.sqrt((n - i - 2 + perPoint**2) * (n - i )))
        # Mask that value and proceed
        xm[minds[-1]] = ma.masked
    # Find the number of outliers
    ofound = False
    for i in range(maxOLs-1, -1, -1):
        if R[i] > L[i]:
            ofound = True
            break
    # Prepare return value
    if ofound:
        return minds[0:i + 1]    # There are outliers
    else:
        return []    # No outliers could be detected

def summarization_nonK_PSM(df, inputDict,keyValue_col, params):
    set_1 = params["set_1"].split(",")
    set_2 = params["set_2"].split(",")
    set_3 = params["set_3"].split(",")
   
    resDict = {}
    reporters = params["tmt_reporters_used"].split(";")
    progress = progressBar(len(inputDict))
    for entry, psms2 in inputDict.items():
        progress.increment()
        psms = df[keyValue_col][df[keyValue_col].isin(psms2)].unique()
        if len(psms) == 0:
            continue
        else:
            subDf = (df[df[keyValue_col].isin(psms)]).drop_duplicates()
            if subDf.rescue_status.isnull().sum() > 1:
                subDf = subDf[subDf.rescue_status.isnull()]
            subDf = (subDf[reporters]).drop_duplicates()
            threshold = 100
            if len(subDf) > threshold:
                psmSum = subDf.sum(axis=1)
                topIndex = psmSum.sort_values(ascending=False).index[:threshold]
                subDf = subDf.loc[topIndex]
               
            subDf = np.log2(subDf)
            psmMeans = subDf.mean(axis=1)
            subDf = subDf.sub(psmMeans, axis=0)

            # Outlier removal
            if len(subDf) >= 3:
                subDf = outlierRemoval(subDf, 0.05, reporters)  # Can I make it faster?
                
            # Protein-level quantification (as a dictionary)
            if len(subDf) > 0:
                subDf = 2 **subDf
                subDf = light_perc(subDf, set_1)
                subDf = light_perc(subDf, set_2)
                subDf = light_perc(subDf, set_3)
                subDf = subDf.mean(axis = 0).replace(np.nan, 0)
                resDict[entry] = subDf.to_dict()

    res = pd.DataFrame.from_dict(resDict, orient="index",columns = reporters)
    return res

def summarization(df, inputDict, keyValue_col, progress_bar, params):
    resDict = {}
    reporters = params["tmt_reporters_used"].split(";")
    progress = progressBar(len(inputDict))
    for entry, psms2 in inputDict.items():
        if progress_bar == 1:
            progress.increment()
        psms = df[keyValue_col][df[keyValue_col].isin(psms2)].unique()
        if len(psms) == 0:
            continue
        else:
            subDf = (df[df[keyValue_col].isin(psms)]).drop_duplicates()
            if subDf.rescue_status.isnull().sum() > 0:
                subDf   = subDf[subDf.rescue_status.isnull()]
            subDf = subDf.reset_index(drop =True)
            # Outlier removal
            subDf[fully_heavy_reporters[0]] =1
            if len(subDf) >= 3:
                subDf = outlierRemoval(subDf, 0.05, reporters)# Can I make it faster?
                
            if len(subDf) > 0:
                subDf2 = subDf[reporters].mean(axis = 0).replace(np.nan, 0).to_frame().T
                if (subDf.rescue_status.notnull().sum() > 0): 
                    subDf2['rescue_status'] = 'rescued_1'
                else:
                    subDf2['rescue_status'] = np.nan
                resDict[entry] = subDf2.iloc[0]
    reporters_temp = reporters.copy(); reporters_temp.append('rescue_status')
    res = pd.DataFrame.from_dict(resDict, orient="index",columns = reporters_temp)
    return res    

def summarization_indReporters(df, inputDict, keyValue_col, params):
    resDict     = {}
    reporters   = params["tmt_reporters_used"].split(";")
    progress = progressBar(len(inputDict))
    for entry, psms2 in inputDict.items():
        progress.increment()
        psms            = df[keyValue_col][df[keyValue_col].isin(psms2)].unique()
        subDf_reporter_ = pd.DataFrame(columns = reporters)
        if len(psms) == 0:
            continue
        else:
            subDf = (df[df[keyValue_col].isin(psms)]).drop_duplicates()
            if subDf.rescue_status.isnull().sum() > 0:
                subDf   = subDf[subDf.rescue_status.isnull()]
        
            subDf = subDf.reset_index(drop=True)
            subDf[fully_heavy_reporters[0]] =1
        if params["standard_outlier"] == "1":  ## Outlier removal with generalised ESD and dixon Q test
            for reporter in reporters:
                subDf_reporter = subDf[[reporter]].reset_index(drop =True)
                subDf_reporter_[reporter] = np.mean(outlierRemoval(subDf_reporter.copy(), 0.05, reporter)).values
        if len(subDf_reporter_) > 0:
            if (subDf.rescue_status.notnull().sum() > 0): 
                subDf_reporter_['rescue_status'] = 'rescued_1'
            else:
                subDf_reporter_['rescue_status'] = np.nan
            resDict[entry] = subDf_reporter_.iloc[0]
    reporters_temp = reporters.copy(); reporters_temp.append('rescue_status')
    res = pd.DataFrame.from_dict(resDict, orient="index",columns = reporters_temp)
    return res

def summarization_basedOn_nonK_prot(df, inputDict, psm2prot_nonK, params):   
    resDict         = {}
    reporters       = params["tmt_reporters_used"].split(";")
    reporters_temp  = params["tmt_reporters_used"].split(";")
    reporters_temp.append('entry')
    reporters_temp.append("normalization_by")
    reporters_temp.append("normalized_status")
    set_1           = params["set_1"].split(",")
    set_2           = params["set_2"].split(",")
    set_3           = params["set_3"].split(",")
    progress        = progressBar(len(inputDict))
    
    for entry, psms2 in inputDict.items():
        progress.increment()
        psms = df['Outfile'][df['Outfile'].isin(psms2)].unique()
        if len(psms) == 0:
            continue
        else:
            subDf3 = (df[(df['Outfile'].isin(psms)) & ((df['Peptide'].isin([entry])) | (df['Protein'].isin([entry]))) ]).drop_duplicates()
            subDf3['average'] = subDf3[reporters].mean(axis=1)
            subDf3 = subDf3.sort_values(by='average',ascending = False).head(np.int(params["PSM_selection"])).reset_index(drop=True)
            
            subDf3 = light_perc(subDf3, set_1)
            subDf3 = light_perc(subDf3, set_2)
            subDf3 = light_perc(subDf3, set_3)
            for prot in subDf3['Protein'].unique():
                subDf= pd.DataFrame(); normalized_status = 1   
                if params["normalization"] == "2":
                    abs_prot = psm2prot_nonK[reporters][psm2prot_nonK.index.isin([prot])].replace(0, np.nan) 
                    if abs_prot.shape[0] > 0:
                        subDf = subDf.append(subDf3[reporters][subDf3['Protein'].isin([prot])]/abs_prot.values)#.reset_index(inplace = True)
                    else:
                        subDf = subDf3[reporters]
                        normalized_status = 0
                else:
                    print("  WARNING: Parmeter 'normalization' is not defined; Check once again")
                    subDf = subDf3[reporters]
                    normalized_status = 0
                
                subDf = subDf.reset_index(drop=True)        
                subDf[fully_heavy_reporters[0]] =1 # seeting it to 1 , so thta this sample will not be considered for outlier removal      
                if params["standard_outlier"] == "1":  ## Outlier removal with generalised ESD and dixon Q test
                    # Outlier removal
                    if len(subDf) >= 3:
                        subDf = outlierRemoval(subDf, 0.05,reporters)  # Can I make it faster?
                
                # summarize the psm ratio to protein
                subDf = pd.DataFrame(subDf.mean(axis = 0)).transpose() 
                      
                if normalized_status == 1:
                    subDf["entry"] = entry
                    subDf["normalization_by"] = prot
                    subDf["normalized_status"] = "yes"
                    key = entry+'_'+prot
                else:
                    subDf["entry"] = entry
                    subDf["normalization_by"] = np.nan
                    subDf["normalized_status"] = "No"
                    key = entry+'_'
                resDict[key] = subDf.values.flatten()
            
    res = pd.DataFrame.from_dict(resDict, orient="index",columns = reporters_temp)           
    return res   


def summarize_heavy_light_psm_to_pep(psm_K1_light, psm_K1_heavy, params):
    
    print("\n  Summarizing heavy and light-PSMs to peptide light-Lys% (L/(L+H))")
    fully_light_reporters   = params["fully_light_reporters"].split(";")
    fully_heavy_reporters   = params["fully_heavy_reporters"].split(";")
    reporters               = params["tmt_reporters_used"].split(";")
    threshold               = np.int(params["PSM_selection"])
    psm_light   = pd.DataFrame()
    psm_heavy   = pd.DataFrame()
    pep_L_perc  = pd.DataFrame()
    progress    = progressBar(len(psm_K1_heavy['Peptide'].unique()))
       
    for pep in psm_K1_heavy['Peptide'].unique():
        progress.increment()
        pep_L_perc_ = pd.DataFrame()
        pep_        = pep.replace('&', '')
        light_psm   = psm_K1_light[psm_K1_light['Peptide'].isin([pep_])].drop_duplicates('Outfile').reset_index(drop = True)
        heavy_psm   = psm_K1_heavy[psm_K1_heavy['Peptide'].isin([pep ])].drop_duplicates('Outfile').reset_index(drop = True)
            
        if light_psm.rescue_status.isnull().sum() > 0:
            light_psm   = light_psm[light_psm.rescue_status.isnull()]
            psmSum      = light_psm[reporters].mean(axis=1)
            topIndex    = psmSum.sort_values(ascending=False).index[:threshold]
            light_psm   = light_psm.loc[topIndex].reset_index(drop = True)
        else:
            psmSum      = light_psm[reporters].mean(axis=1)
            topIndex    = psmSum.sort_values(ascending=False).index[:threshold]
            light_psm   = light_psm.loc[topIndex].reset_index(drop = True)
            
        if heavy_psm.rescue_status.isnull().sum() > 0:
            heavy_psm   = heavy_psm[heavy_psm.rescue_status.isnull()]
            psmSum      = heavy_psm[reporters].mean(axis=1)
            topIndex    = psmSum.sort_values(ascending=False).index[:threshold]
            heavy_psm   = heavy_psm.loc[topIndex].reset_index(drop = True)
        else:
            psmSum      = heavy_psm[reporters].mean(axis=1)
            topIndex    = psmSum.sort_values(ascending=False).index[:threshold]
            heavy_psm   = heavy_psm.loc[topIndex].reset_index(drop = True)
        
        # calculate the the fraction of each channel to the total
        light_psm[reporters]             = light_psm[reporters].div(light_psm[reporters].sum(axis=1), axis=0)
        heavy_psm[reporters]             = heavy_psm[reporters].div(heavy_psm[reporters].sum(axis=1), axis=0)
        light_psm[fully_heavy_reporters] = 0
        heavy_psm[fully_light_reporters] = 0
            
        #Calculate  the heavy (H) and light (L) peptide ratio at MS1 intesity level
        ligh_and_heavy_psm = light_psm.append(heavy_psm).reset_index(drop = True)
        if ligh_and_heavy_psm.rescue_status.isnull().sum() > 0:
            ligh_and_heavy_psm   = ligh_and_heavy_psm[ligh_and_heavy_psm.rescue_status.isnull()]
        
        H_L_ratio_MS1int        = ligh_and_heavy_psm[['H_L_ratio']].reset_index(drop =True)
        H_L_ratio_MS1int_filter = H_L_ratio_MS1int.H_L_ratio.isin([0, np.nan, np.inf, -np.inf])
        H_L_ratio_MS1int        = H_L_ratio_MS1int[~H_L_ratio_MS1int_filter]
            
        if ((len(light_psm) > 0)  & (len(heavy_psm) > 0)) & (len(H_L_ratio_MS1int) > 0) :
            if len(H_L_ratio_MS1int) >= 3:
                H_L_ratio_MS1int = outlierRemoval(H_L_ratio_MS1int.copy(), 0.05, ['H_L_ratio'])
        
            light_psm[reporters] = light_psm[reporters]*(H_L_ratio_MS1int['H_L_ratio'].mean())
            light_psm            = light_psm[(light_psm[reporters] >= 0).all(axis=1)]
            
            if len(light_psm) > 0:
                #summarize light-PSM to peptide
                psm2pep = light_psm.groupby('Peptide')['Outfile'].apply(list).to_dict()
                keyValue_col = 'Outfile'
                if params['outlier_removal_stratagy'] == '2':
                    psm2pep_light = summarization_indReporters(light_psm, psm2pep,keyValue_col, params)
                else:
                    progress_bar  = 0
                    psm2pep_light = summarization(light_psm, psm2pep,keyValue_col, progress_bar, params)
                    
                #summarize heavy-PSM to peptide
                psm2pep = heavy_psm.groupby('Peptide')['Outfile'].apply(list).to_dict()
                if params['outlier_removal_stratagy'] == '2':
                    psm2pep_heavy = summarization_indReporters(heavy_psm, psm2pep,keyValue_col, params)
                else:
                    progress_bar  = 0
                    psm2pep_heavy = summarization(heavy_psm, psm2pep,keyValue_col, progress_bar, params)
                    
                #calculate light-Lys pepcenatge  (L/(L+H))
                pep_L_perc_ = (psm2pep_light[reporters]).div(psm2pep_light[reporters].add(psm2pep_heavy[reporters].values, fill_value=0))
                    
                #assign the rescue status
                if (psm2pep_light.rescue_status.notnull().sum() > 0) | (psm2pep_heavy.rescue_status.notnull().sum() > 0): 
                    pep_L_perc_['rescue_status'] = 'rescued_1'
                else:
                    pep_L_perc_['rescue_status'] = np.nan
            else:
                heavy_psm = pd.DataFrame() # setting heavy PSM to null because the there are no corresponding light PSMs
        if len(pep_L_perc_) > 0:
            pep_L_perc = pd.concat([pep_L_perc,pep_L_perc_],axis=0)
            psm_light = pd.concat([psm_light,light_psm],axis=0)
            psm_heavy = pd.concat([psm_heavy,heavy_psm],axis=0)
                
    return pep_L_perc, psm_light, psm_heavy

def define_noise_in_each_PSM(row):
    if ((row['noise_channel'] == 0.5*row['s_1lowest']) and ((row['s_2lowest'] - row['s_1lowest']) > 0.5*row['s_min'])):
        return float(row['nclevel'])*row['noise_channel']
    elif ((row['noise_channel'] == 0.5*row['s_1lowest']) and 0.5*((row['s_2lowest'] - row['s_1lowest']) < row['s_min'])):
        return float(row['nclevel'])*(row['s_2lowest'] -row['s_min'])
    else:
        return float(row['nclevel'])*(row['s_1lowest'] - row['s_min'])
def make_directory_prot_output(params):
    if os.path.exists(os.path.join(os.getcwd(), params["output_folder_protein"])):
        i = 1
        while os.path.exists(os.path.join(os.getcwd(),  params["output_folder_protein"]+'_%s' %i)):
            i += 1
        saveDir_prot = os.path.join(os.getcwd(),  params["output_folder_protein"]+'_%s' %i)
        os.makedirs(saveDir_prot, exist_ok=True)
    else: 
        saveDir_prot = os.path.join(os.getcwd(),  params["output_folder_protein"])
        os.makedirs(os.path.join(os.getcwd(), params["output_folder_protein"]))
    return saveDir_prot
def make_directory_lysine_output(params):
    if os.path.exists(os.path.join(os.getcwd(), params["output_folder_lysine"])):
        i = 1
        while os.path.exists(os.path.join(os.getcwd(),  params["output_folder_lysine"]+'_%s' %i)):
            i += 1
        saveDir_lys = os.path.join(os.getcwd(),  params["output_folder_lysine"]+'_%s' %i)
        os.makedirs(saveDir_lys, exist_ok=True)
    else: 
        saveDir_lys = os.path.join(os.getcwd(),  params["output_folder_lysine"])
        os.makedirs(os.path.join(os.getcwd(), params["output_folder_lysine"]))
    return saveDir_lys
# In[0]: 
##################################################  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   #########################################################################
if __name__ == "__main__":
    print(" currently running the SILAC-TMT data processing pipeline (silac_tmt_v1.0.0) \n")
    for no_arg in range(1,len(sys.argv)):
    ###########################################
    # Read parameter file                      #
    ###########################################
        paramFile = sys.argv[no_arg]
        params = getParams(paramFile)
        print("  parameters used in this program\n")
        print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in params.items()) + "}")
        
        reporters = params["tmt_reporters_used"].split(";")
        fully_light_reporters = params["fully_light_reporters"].split(";")
        fully_heavy_reporters = params["fully_heavy_reporters"].split(";")
        
        ###########################################
        # create a directory                      #
        ########################################### 
        saveDir_prot = make_directory_prot_output(params)
        
        
        ####################################################################################################
        # JUMP-q (quantification) of PSMs from ID.txt                                                      #
        ########################################### #######################################################
        if 'idtxt' in params:
            print("  Excuting JUMPq for the quantification of ID.txt for batch no. %d \n" %no_arg)
            ##################
            # Parsing ID.txt #
            ##################
            print("  Quantifying proteins for batch no. %d \n" %no_arg)
            print("  Loading ID.txt file for batch no. %d \n" %no_arg)
            dfId = pd.read_table(params["idtxt"], sep=";", skiprows=1, header=0)
            
            ##### Cleaning undesired character combinations in double Peptide sequqnce ############################
            letter = '&'
            letter2 = 'K'
            for index in dfId.index:
                    if letter in dfId.loc[index, 'Peptide']:
                         string = dfId.loc[index, 'Peptide']
                         if(string[string.find(letter)-1]!=letter2 or string[string.rfind(letter)-1]!=letter2):
                              string2=re.sub(r'&', "", string)
                              string2=re.sub(r'K', "K&", string2)
                              string2=re.sub(r'K&\.', "K.", string2)
                              string2=re.sub(r'\.K&', ".K", string2)
                              string2=re.sub(r'K\.K', "K.K&", string2)
                              dfId.loc[index, 'Peptide']=string2           
            ########################################################################################################
            #handling ID.txt
            dfId["frac"] = dfId["Outfile"].apply(lambda x: os.path.dirname(x).rsplit(".", 1)[0] + ".mzXML")
            dfId["scan"] = dfId["Outfile"].apply(lambda x: os.path.basename(x).split(".")[1])
            dfId["key"]  = dfId["frac"] + "_" + dfId["scan"]
            fracs = dfId["frac"].unique()
    
            ##################################
            # Extract TMT reporter ion peaks #
            ##################################
            # 1st round of reporter ion extraction
            dfQuan, reporterSummary = extractReporters(fracs, dfId, params)
        
            # Before 2nd round of TMT reporter extraction, m/z-shifts of reporters are summarized
            print("\n  m/z-shift in each TMT reporter")
            reporters = params["tmt_reporters_used"].split(";")
            for reporter in reporters:
                m = reporterSummary[reporter]["meanMzShift"]
                s = reporterSummary[reporter]["sdMzShift"]
                print("    %s\tm/z-shift = %.4f [ppm]\tsd = %.4f" % (reporter, m, s))
        
            # 2nd round of reporter ion extraction
            dfQuan, reporterSummary = extractReporters(fracs, dfId, params, **reporterSummary)
    
            ###################################################################################################
            # get the min intesity in complete batch and assign to PSMs where the reporter are not quantified #
            ###################################################################################################
            reporters = params["tmt_reporters_used"].split(";")
            dfQuan_min = np.nanmin(dfQuan[reporters].replace(0, np.nan).values)
            dfQuan['rescue_status'] = np.nan
            dfQuan.loc[(dfQuan[reporters] < dfQuan_min).any(axis=1),'rescue_status']='rescued_0'
            dfQuan[reporters] = dfQuan[reporters].apply(lambda x: np.where(x < dfQuan_min, dfQuan_min, x))
            dfQuan.to_csv(os.path.join(saveDir_prot, "quan_psm_raw.txt"), sep="\t")
            ###########################
            # TMT impurity correction #
            ###########################
            dfQuan = correctImpurity(dfQuan, params)
            
            ################################################################
            # Assign the value 1 to the reporter  where the  intesity <= 0 #
            ################################################################
            dfQuan[reporters] = dfQuan[reporters].apply(lambda x: np.where(x < 1, 1, x))
            
            #########################################
            # Assign the protein and peptide to PSM #
            #########################################
            
            dfId2 = dfId[['Peptide', 'Protein', 'Outfile','key']]#.set_index('key')
            dfQuan_ = dfId2.merge(dfQuan, left_on='key', right_on=dfQuan.index) 
            dfQuan_.to_csv(os.path.join(saveDir_prot, "quan_psm_impurity_corrected.txt"), sep="\t")
            psmQuan = dfQuan_.copy()
            psmQuan["frac"] = psmQuan["Outfile"].apply(lambda x: os.path.dirname(x).rsplit(".", 1)[0] + ".mzXML")
            psmQuan["scan"] = psmQuan["Outfile"].apply(lambda x: os.path.basename(x).split(".")[1])
        else:
            psmQuan = pd.read_csv(params["psm_quant"],  sep="\t", skiprows=0,index_col=0 ).reset_index(drop =True)
            psmQuan["frac"] = psmQuan["Outfile"].apply(lambda x: os.path.dirname(x).rsplit(".", 1)[0] + ".mzXML")
            psmQuan["scan"] = psmQuan["Outfile"].apply(lambda x: os.path.basename(x).split(".")[1])
            
    
    
        #########################################
        # SILAC-TMT data processing             #
        ######################################## 
    
        psmQuan['Peptide_only'] = psmQuan['Peptide'].str.split('.').str[1]
        psmQuan['count_K'] = psmQuan['Peptide_only'].str.count('K')
        psm_K1 = psmQuan.loc[(psmQuan['count_K'] == 1),:]
        psm_K2 = psmQuan.loc[(psmQuan['count_K'] == 2),:]
        psm_nonK = psmQuan.loc[(psmQuan['count_K'] == 0),:]
        print("  Summary of PSMs:\n  # of total PSMs = %d\n  # of non-Lys PSMs =%d \n  # of single-Lys PSMs = %d \n  # of double-Lys PSMs = %d \n" %( len(psmQuan), len(psm_nonK), len(psm_K1), len(psm_K2)))
    
        id_all_pep = pd.read_table(params["id_res_folder"] + '/id_uni_pep.txt', sep="\t", skiprows=3, header=0, index_col = 0)
        id_all_pep = id_all_pep[['Protein Group#', 'Protein Accession #', 'Protein Description', 'GN',]]
                                                 
        id_uni_prot = pd.read_table(params["id_res_folder"] + '/id_uni_prot.txt', sep="\t", skiprows=1, header=0, index_col = 1)
        id_uni_prot = id_uni_prot[['Protein Group#', 'Protein Description', 'GN',]]
            
        id_all_prot = pd.read_table(params["id_res_folder"] + '/id_all_prot.txt', sep="\t", skiprows=1, header=0, index_col = 1)
        id_all_prot = id_all_prot[['Protein Group#', 'Protein Description', 'GN',]]
                                   
        
        avgBias, sdBias, semBias, nn = getLoadingBias(psm_nonK, params)
        print("  Loading bias (before correction)")
        print("  Reporter\tMean[%]\tSD[%]\tSEM[%]\t#PSMs")
        for i in range(len(reporters)):
            print("  %s\t%.2f\t%.2f\t%.2f\t%d" % (reporters[i], avgBias[i], sdBias[i], semBias[i], nn))
        # Normalization ( Loading bias correction) #
        if params["loading_bias_correction"]  == "1":
            psm_nonK, psm_K1 = normalization_Lys_PSM(psm_nonK, psm_K1, params)
            print("\n  Loading Bias correction performed for the all the channels in non-Lys PSMs and single-K PSMs")
            
    # In[]  
        #######################################################
        # delete PSM with decoys, contaminats, and wierd cases#
        #######################################################
        psm_K1 = psm_K1[~psm_K1.Protein.str.contains("co|CON_")]
        psm_K1 = psm_K1[~psm_K1.Protein.str.contains("Decoy_")]
        psm_K1 = psm_K1[~psm_K1.Peptide.str.contains("&K")]
        psm_K1 = psm_K1[~psm_K1.Peptide.str.contains("R&")]
        psm_K1 = psm_K1.reset_index(drop = True) 
    
        ##########################################################################################
        # Noise removal to reduce ratio compression in TMT  using 0-day and SILAM quantification
        #########################################################################################
        #define the minimum intesity observed in the whole batch
        psm_K1['s_min'] = np.nanmin(psm_K1[reporters].replace(0, np.nan).values)
        psm_K1.loc[:, ['heavy_light']] = psm_K1.loc[:, 'Peptide_only'].str.count('&')
        psm_K1_heavy = psm_K1.loc[(psm_K1['heavy_light'] == 1),:]
        psm_K1_light = psm_K1.loc[(psm_K1['heavy_light'] == 0),:]
        psm_K1_heavy.drop(columns=['heavy_light', 'count_K', 'Peptide_only'],axis=1,inplace=True)
        psm_K1_light.drop(columns=['heavy_light', 'count_K', 'Peptide_only'],axis=1,inplace=True)
    
        reporters_temp = reporters.copy()
        reporters_temp.append('noise_channel')
        if params["noise_removal_in_heavyPSM"]  == "1":
            psm_K1_heavy.loc[(psm_K1_heavy[fully_light_reporters].min(axis=1) > (psm_K1_heavy[fully_heavy_reporters].min(axis=1))),'rescue_status']='rescued_1'
            psm_K1_heavy['noise_channel'] = psm_K1_heavy.apply(lambda row: row[fully_light_reporters].min() if row[fully_light_reporters].min() <= (row[fully_heavy_reporters].min()) else (row[fully_heavy_reporters].min()) ,axis=1)
            psm_K1_heavy['s_1lowest'] = (psm_K1_heavy[reporters_temp]).min(axis=1)
            psm_K1_heavy['s_2lowest'] = psm_K1_heavy.loc[:, (reporters)].apply(lambda row: row.nsmallest(2).values[-1],axis=1)
            psm_K1_heavy['nclevel'] = params["nc_level"]
            psm_K1_heavy.loc[~((psm_K1_heavy['noise_channel'] == psm_K1_heavy['s_1lowest']) & ((psm_K1_heavy['s_2lowest'] - psm_K1_heavy['s_1lowest']) > psm_K1_heavy['s_min']) ),'rescue_status']='rescued_1'
            psm_K1_heavy['tmt_noise'] = psm_K1_heavy.apply(define_noise_in_each_PSM,axis=1)
    
            d_heavy = {cl: lambda x, cl=cl: x[cl] - x['tmt_noise']*np.float(params["NoiseThreshold_completely_unlabeled"]) for cl in reporters}
            psm_K1_heavy = psm_K1_heavy.assign(**d_heavy)
            print("\n  TMT noise correction has been performed for heavy PSMs by the completely unlabeled channel (eg. day-0) quantification")
            
        # TMT noise correction in light PSM by SILAM quant
        if params["noise_removal_in_lightPSM"]  == "1":
            psm_K1_light.loc[(psm_K1_light[fully_heavy_reporters].min(axis=1) > (psm_K1_light[fully_light_reporters].min(axis=1))),'rescue_status']='rescued_1'
            psm_K1_light['noise_channel'] = psm_K1_light.apply(lambda row: row[fully_heavy_reporters].min() if row[fully_heavy_reporters].min() <= (row[fully_light_reporters].min()) else (row[fully_light_reporters].min()) ,axis=1)
    
            psm_K1_light['s_1lowest'] = (psm_K1_light[reporters_temp]).min(axis=1)
            psm_K1_light['s_2lowest'] = psm_K1_light.loc[:, (reporters)].apply(lambda row: row.nsmallest(2).values[-1],axis=1)
            psm_K1_light['nclevel'] = params["nc_level"]
            psm_K1_light.loc[~((psm_K1_light['noise_channel'] == psm_K1_light['s_1lowest']) & ((psm_K1_light['s_2lowest'] - psm_K1_light['s_1lowest']) > psm_K1_light['s_min']) ),'rescue_status']='rescued_1'
            psm_K1_light['tmt_noise'] = psm_K1_light.apply(define_noise_in_each_PSM,axis=1)
            d_light = {cl: lambda x, cl=cl: x[cl] - x['tmt_noise']*np.float(params["NoiseThreshold_completely_labeled"]) for cl in reporters}
            psm_K1_light = psm_K1_light.assign(**d_light)
            psm_K1_light['norm_state'] = 'No'
            print("\n  TMT noise correction has been performed for light PSMs by the completely labeled channel (eg. SILAM) quantification")
        
        
                
        # In[3]:
        if params["normalization"]  == "2":
            print("\n  non-Lys based protein abundance is sected for normalization \n")
            #######################################################################################################################  
            #summarize non-lys PSM to proteins  (includes outlier removal steps) and calculates protein abundance from non-Lys PSM
            ######################################################################################################################
            if 'prot_absQuna_byNonKpsm' in params:
                psm2prot_nonK = pd.read_csv(params['prot_absQuna_byNonKpsm'] ,  sep='\t', index_col=0) 
            else:
                print("\n  Summarizing non-lys PSMs to proteins  (includes outlier removal steps)\n")
                prot2nonK_psm = psm_nonK.groupby('Protein')['Outfile'].apply(list).to_dict()
                keyValue_col = 'Outfile'
                psm2prot_nonK = summarization_nonK_PSM(psm_nonK, prot2nonK_psm, keyValue_col, params)
                
                outfile = os.path.join(saveDir_prot, 'prot_absQuna_byNonKpsm.txt')
                psm2prot_nonK.to_csv(outfile, sep='\t', mode='w') 
                
            # In[4]: 
            ############################################################################
            #Summarize light PSM ratio to protein level (includes outlier removal steps) 
            ############################################################################
        
            print("  Summarizing light PSM ratio to proteins and peptide (includes outlier removal steps)")
            if params["standard_outlier"] == "1":
                print("\n  Outlier PSMs will be removed by generalised ESD and dixon Q-test")
            else:
                print("\n  Outlier PSMs will not be removed by generalised ESD and dixon Q-test")
                    
            print("\n  Summarization of Proteins with non-Lys based normalization method")
            prot2psm = psm_K1_light.groupby('Protein')['Outfile'].apply(list).to_dict()
            psm2prot_1K = summarization_basedOn_nonK_prot(psm_K1_light, prot2psm, psm2prot_nonK, params)
            psm2prot_1K.set_index('entry', inplace = True)
            #Saving the output
            print("  Proteins with turnover data in the current batch  = %d " %(len(psm2prot_1K)))
            psm2prot_1K = pd.merge(psm2prot_1K, id_all_prot, left_index=True, right_index=True,how="left") #right_on='CustomerID',
            
            outfile = open(os.path.join(saveDir_prot, 'all_protein_turnover.txt'),"w")
            outfile.write("All proteins quantified (n = %d)\n" %psm2prot_1K.shape[0] )
            psm2prot_1K.index.name = "Protein Accession"
            psm2prot_1K.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
            outfile.close()
    
            #filtering unique protein and sving to outfile
            psm2prot_1K = psm2prot_1K[psm2prot_1K.index.isin(id_uni_prot.index)]
        
            outfile = open(os.path.join(saveDir_prot, 'uni_protein_turnover.txt'),"w")
            outfile.write("Unique proteins quantified (n = %d)\n" %psm2prot_1K.shape[0] )
            psm2prot_1K.index.name = "Protein Accession"
            psm2prot_1K.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
            outfile.close()
            
            # In[5]:    
            #############################################
            # Summarize light PSM ratio to peptide level
            ############################################
            
            print("\n   Summarization of peptides with non-Lys based normalization method")
            pep2psm     = psm_K1_light.groupby('Peptide')['Outfile'].apply(list).to_dict()
            psm2pep_1K = summarization_basedOn_nonK_prot(psm_K1_light, pep2psm, psm2prot_nonK, params)
            psm2pep_1K.set_index('entry', inplace = True)
    
            #Saving the output
            print("  Peptides with turnover data in the current batch  = %d " %(len(psm2pep_1K)))
            outfile = open(os.path.join(saveDir_prot, 'peptide_turnover.txt'),"w")
            outfile.write("peptides quantified (n = %d)\n" %psm2pep_1K.shape[0] )
            psm2pep_1K.index.name = "Peptide"
            psm2pep_1K.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
            outfile.close()
            
        else: #if params["normalization"]  == "1":
            print("\n  MS1 quantification based normalization is selected")
            ##########################################################################################
            # calculating peptide isotopic distribution  and  extract their MS1 precursor intesnities
            #########################################################################################
            if 'ms1_quant' in params:
                dfQuan_ms1 = pd.read_csv(params['ms1_quant'],  sep="\t", skiprows=0,index_col=0 )
            else:
                if 'pepIsoDist' in params:
                    pepIsoDist = pd.read_csv(params['pepIsoDist'],  sep="\t", skiprows=0,index_col=0 )
                else:
                    pepIsoDist = get_isope_distri(psm_K1, params) # calculating peptide isotopic distribution
                    outfile = open(os.path.join(saveDir_prot, 'pepIsoDist'),"w")
                    pepIsoDist.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
                    
                fracs = psm_K1['frac'].unique()
                print("\n   Extracting the MS1 intensities\n ")
                progress = progressBar(len(fracs))
                dfQuan_ms1 = pd.DataFrame()
                for file in fracs:
                    print("\n   Extracting the MS1 intensities from {}".format(os.path.basename(file)))        
                    progress.increment()
                    df = psm_K1[psm_K1['frac'].isin([file])]  
                    dfQuan_ms1_ = extract_MS1_intesnity(file, df, pepIsoDist, params);  #extract  MS1 precursor intesnities
                    dfQuan_ms1  = pd.concat([dfQuan_ms1,dfQuan_ms1_],axis=0)
    
                outfile = open(os.path.join(saveDir_prot, 'ms1quan.txt'),"w")
                dfQuan_ms1.index.name = "scan"
                dfQuan_ms1.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
            
            psm_K1_light = psm_K1_light.merge(dfQuan_ms1, left_on='key', right_on=dfQuan_ms1.index) 
            psm_K1_light['H_L_ratio'] = psm_K1_light['light_intensity']/psm_K1_light['heavy_intensity']; 
             
            psm_K1_heavy = psm_K1_heavy.merge(dfQuan_ms1, left_on='key', right_on=dfQuan_ms1.index) 
            psm_K1_heavy['H_L_ratio'] = psm_K1_heavy['light_intensity']/psm_K1_heavy['heavy_intensity']; 
    
            ##########################################################################################
            # Calculate the Light-Lys % in each peptide 
            #step-1: Summarize the light and heavy PSM to light and heavy peptide 
            #step-2 calculate the light-Lys%,  L/(L+H) in each peptde using heavy and light peptide
            #########################################################################################
            pep_L_perc, psm_light, psm_heavy = summarize_heavy_light_psm_to_pep(psm_K1_light, psm_K1_heavy, params)
    
    
            prot2lightPep = psm_K1_light[['Peptide', 'Protein']].drop_duplicates().set_index('Peptide')
            psm_light = psm_light.drop('Protein', axis=1)
            psm_light = pd.merge(left=psm_light, right=prot2lightPep, how="left", left_on='Peptide', right_index=True) #left_on='Protein', right_on='Protein Accession #'
    
            prot2heavyPep = psm_K1_heavy[['Peptide', 'Protein']].drop_duplicates().set_index('Peptide')
            psm_heavy = psm_heavy.drop('Protein', axis=1)
            psm_heavy = pd.merge(left=psm_heavy, right=prot2heavyPep, how="left", left_on='Peptide', right_index=True) #left_on='Protein', right_on='Protein Accession #'
    
            print("  # of Peptides (with L & H pairs) turnover data in the current batch  = %d " %(len(pep_L_perc)))
            pep_L_perc = pd.merge(left=pep_L_perc, right=prot2lightPep, how="left", left_index=True, right_index=True) #left_on='Protein', right_on='Protein Accession #'
            pep_L_perc.index.name = "Peptide"
            pep_L_perc.reset_index(inplace=True)
            
            outfile = open(os.path.join(saveDir_prot, 'peptide_turnover.txt'),"w")
            outfile.write("peptides quantified (n = %d)\n" %pep_L_perc.shape[0] )
            pep_L_perc.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')#,index = None
            outfile.close()
            
            if params['summarize_protein_from_raw_data'] == '2':
                print("\n  Summarizing the psm to protein (L%)")
                
                print("\n  step-1: Summarizing the light psm to proteins")
                psm2prot = psm_light.groupby('Protein')['Outfile'].apply(list).to_dict()
                keyValue_col = 'Outfile'
                if params['outlier_removal_stratagy'] == '2':
                    psm2prot_light = summarization_indReporters(psm_light, psm2prot, keyValue_col, params)
                else:
                    progress_bar   = 1
                    psm2prot_light = summarization(psm_light, psm2prot, keyValue_col,progress_bar, params)
                psm2prot= psm_heavy.groupby('Protein')['Outfile'].apply(list).to_dict()
                
                print("\n  step-2: Summarizing the heavy psm to proteins")
                if params['outlier_removal_stratagy'] == '2':
                    psm2prot_heavy = summarization_indReporters(psm_heavy, psm2prot, keyValue_col, params)
                else:
                    progress_bar   = 1
                    psm2prot_heavy = summarization(psm_heavy, psm2prot, keyValue_col, progress_bar, params)
                
                # make sure that protein has both light and heavy ; filter the proteins which has only light or heavy
                if len(psm2prot_light) > len(psm2prot_heavy):
                    psm2prot_light = psm2prot_light[psm2prot_light.index.isin(psm2prot_heavy.index)]
                if len(psm2prot_light) < len(psm2prot_heavy):
                    psm2prot_heavy = psm2prot_heavy[psm2prot_heavy.index.isin(psm2prot_light.index)]
                
                print("\n  step-3: Calculating light-K %  (L/(L+H)) in each protein ")
                prot_L_percent = psm2prot_light.copy()
                prot_L_percent[reporters]= (psm2prot_light[reporters]).div(psm2prot_light[reporters].add(psm2prot_heavy[reporters].values, fill_value=0)).values
                
                prot_L_percent = pd.merge(prot_L_percent, id_all_prot, left_index=True, right_index=True,how="left") 
                outfile = open(os.path.join(saveDir_prot, 'all_protein_turnover.txt'),"w")
                outfile.write("All proteins quantified (n = %d)\n" %prot_L_percent.shape[0] )
                prot_L_percent.index.name = "Protein Accession"
                prot_L_percent.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
                outfile.close()
                
                #filtering unique protein and saving to outfile
                prot_L_percent = prot_L_percent[prot_L_percent.index.isin(id_uni_prot.index)]
                outfile = open(os.path.join(saveDir_prot, 'uni_protein_turnover.txt'),"w")
                outfile.write("# of Unique proteins (with L & H pairs) quantified (n = %d)\n" %prot_L_percent.shape[0] )
                prot_L_percent.index.name = "Protein Accession"
                prot_L_percent.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
                outfile.close()
            else:
                print("\n  Summarizing the peptide (litght-Lys%) to protein(litght-Lys%)")
                psm2prot = pep_L_perc.groupby('Protein')['Peptide'].apply(list).to_dict()
                keyValue_col = 'Peptide'
                if params['outlier_removal_stratagy'] == '2':
                    prot_L_percent = summarization_indReporters(pep_L_perc, psm2prot, keyValue_col,params)
                else:
                    progress_bar   = 1
                    prot_L_percent = summarization(pep_L_perc, psm2prot, keyValue_col, progress_bar, params)
                
                #Saving the output
                print("  # of Proteins (with L & H pairs) turnover data in the current batch  = %d " %(len(prot_L_percent)))
                prot_L_percent = pd.merge(prot_L_percent, id_all_prot, left_index=True, right_index=True,how="left") #right_on='CustomerID',
        
                outfile = open(os.path.join(saveDir_prot, 'all_protein_turnover.txt'),"w")
                outfile.write("# of proteins (with L & H pairs) quantified (n = %d)\n" %prot_L_percent.shape[0] )
                prot_L_percent.index.name = "Protein Accession"
                prot_L_percent.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
                outfile.close()
        
                #filtering unique protein and saving to outfile
                prot_L_percent = prot_L_percent[prot_L_percent.index.isin(id_uni_prot.index)]
                outfile = open(os.path.join(saveDir_prot, 'uni_protein_turnover.txt'),"w")
                outfile.write("# of Unique proteins (with L & H pairs) quantified (n = %d)\n" %prot_L_percent.shape[0] )
                prot_L_percent.index.name = "Protein Accession"
                prot_L_percent.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
        #######################################################
        # Free Lysine estimation using double K peptides#
        #######################################################
            if params["free_lysine_estimation"]  == "1": 
                print("Currently running the free lysine estimation pipeline V1.0.0 \n")
                                   
                saveDir_lys = make_directory_lysine_output(params)
            ################################################################################################
            # Normalization ( Loading bias correction) #
            if params["loading_bias_correction"]  == "1":
                psm_nonK, psm_K2 = normalization_Lys_PSM(psm_nonK, psm_K2, params)
                print("\n  Loading Bias correction performed for the all the channels in non-Lys PSMs and double-K PSMs")
                
            #######################################################
            # delete PSM with decoys, contaminats, and wierd cases#
            #######################################################
            psm_K2 = psm_K2[~psm_K2.Protein.str.contains("co|CON_")]
            psm_K2 = psm_K2[~psm_K2.Protein.str.contains("Decoy_")]
            psm_K2 = psm_K2[~psm_K2.Peptide.str.contains("R&")]
            psm_K2 = psm_K2.reset_index(drop = True) 
        
            ##########################################################################################
            # Noise removal to reduce ratio compression in TMT  using 0-day and SILAM quantification
            #########################################################################################
            #define the minimum intesity observed in the whole batch
            psm_K2['s_min'] = np.nanmin(psm_K2[reporters].replace(0, np.nan).values)
            psm_K2.loc[:, ['heavy_light']] = psm_K2.loc[:, 'Peptide_only'].str.count('&')
            psm_K2 = psm_K2.loc[(psm_K2['heavy_light'] > 0),:]
        
            psm_K2_heavy = psm_K2.loc[(psm_K2['heavy_light'] == 2),:]
            psm_K2_mixed = psm_K2.loc[(psm_K2['heavy_light'] == 1),:]
        
            reporters_temp = reporters.copy()
            reporters_temp.append('noise_channel')
        
            if params["noise_removal_in_heavyPSM"]  == "1":
                psm_K2_heavy.loc[(psm_K2_heavy[fully_light_reporters].min(axis=1) > (psm_K2_heavy[fully_heavy_reporters].min(axis=1))),'rescue_status']='rescued_1'
                psm_K2_heavy['noise_channel'] = psm_K2_heavy.apply(lambda row: row[fully_light_reporters].min() if row[fully_light_reporters].min() <= (row[fully_heavy_reporters].min()) else (row[fully_heavy_reporters].min()) ,axis=1)
                psm_K2_heavy['s_1lowest'] = (psm_K2_heavy[reporters_temp]).min(axis=1)
                psm_K2_heavy['s_2lowest'] = psm_K2_heavy.loc[:, (reporters)].apply(lambda row: row.nsmallest(2).values[-1],axis=1)
                psm_K2_heavy['nclevel'] = params["nc_level"]
                psm_K2_heavy.loc[~((psm_K2_heavy['noise_channel'] == psm_K2_heavy['s_1lowest']) & ((psm_K2_heavy['s_2lowest'] - psm_K2_heavy['s_1lowest']) > psm_K2_heavy['s_min']) ),'rescue_status']='rescued_1'
                psm_K2_heavy['tmt_noise'] = psm_K2_heavy.apply(define_noise_in_each_PSM,axis=1)

                d_heavy = {cl: lambda x, cl=cl: x[cl] - x['tmt_noise']*np.float(params["NoiseThreshold_completely_unlabeled"]) for cl in reporters}
                psm_K2_heavy = psm_K2_heavy.assign(**d_heavy)
                
                # TMT noise correction in mixed labeling PSM by fully-unlabled channel (day-0) quant
                psm_K2_mixed.loc[(psm_K2_mixed[fully_light_reporters].min(axis=1) > (psm_K2_mixed[fully_heavy_reporters].min(axis=1))),'rescue_status']='rescued_1'
                psm_K2_mixed['noise_channel'] = psm_K2_mixed.apply(lambda row: row[fully_light_reporters].min() if row[fully_light_reporters].min() <= (row[fully_heavy_reporters].min()) else (row[fully_heavy_reporters].min()) ,axis=1)
                psm_K2_mixed['s_1lowest'] = (psm_K2_mixed[reporters_temp]).min(axis=1)
                psm_K2_mixed['s_2lowest'] = psm_K2_mixed.loc[:, (reporters)].apply(lambda row: row.nsmallest(2).values[-1],axis=1)
                psm_K2_mixed['nclevel'] = params["nc_level"]
                psm_K2_mixed.loc[~((psm_K2_mixed['noise_channel'] == psm_K2_mixed['s_1lowest']) & ((psm_K2_mixed['s_2lowest'] - psm_K2_mixed['s_1lowest']) > psm_K2_mixed['s_min'])),'rescue_status']='rescued_1'
                psm_K2_mixed['tmt_noise'] = psm_K2_mixed.apply(define_noise_in_each_PSM,axis=1)
		d_mixed = {cl: lambda x, cl=cl: x[cl] - x['tmt_noise']*np.float(params["NoiseThreshold_completely_unlabeled"]) for cl in reporters}
                psm_K2_mixed = psm_K2_mixed.assign(**d_mixed)
             
                print("\n  TMT noise correction has been performed for heavy and mixed labeling PSMs by the completely unlabeled channel (eg. day-0) quantification")
        
            psm_K2_heavy = psm_K2_heavy.copy()
            psm_K2_mixed = psm_K2_mixed.copy()
            psm_K2_heavy.Peptide = psm_K2_heavy.Peptide.str.replace('&', '')
            psm_K2_mixed.Peptide = psm_K2_mixed.Peptide.str.replace('&', '')
        
            psm_K2_heavy = psm_K2_heavy[psm_K2_heavy.Peptide.isin(psm_K2_mixed.Peptide)]
            psm_K2_mixed = psm_K2_mixed[psm_K2_mixed.Peptide.isin(psm_K2_heavy.Peptide)]
        
            ##########################################################################################
            # calculating peptide isotopic distribution  and  extract their MS1 precursor intesnities
            #########################################################################################
            
            pepIsoDist = get_isope_distri_lys(psm_K2, params) # calculating peptide isotopic distribution
            outfile = open(os.path.join(saveDir_lys, 'pepIsoDist'),"w")
            pepIsoDist.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
        
            fracs = psm_K2['frac'].unique()
            print("\n   Extracting the MS1 intensities\n ")
            progress = progressBar(len(fracs))
            dfQuan_ms1 = pd.DataFrame()
            for file in fracs:
                print("\n   Extracting the MS1 intensities from {}".format(os.path.basename(file)))        
                progress.increment()
                df = psm_K2[psm_K2['frac'].isin([file])]  
                dfQuan_ms1_ = extract_MS1_intesnity_for_lys(file, df, pepIsoDist, params);  #extract  MS1 precursor intesnities
                dfQuan_ms1  = pd.concat([dfQuan_ms1,dfQuan_ms1_],axis=0)
        
            outfile = open(os.path.join(saveDir_lys, 'ms1quan.txt'),"w")
            dfQuan_ms1.index.name = "scan"
            dfQuan_ms1.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')
               
            psm_K2_mixed = psm_K2_mixed.merge(dfQuan_ms1, left_on='key', right_on=dfQuan_ms1.index) 
            psm_K2_heavy = psm_K2_heavy.merge(dfQuan_ms1, left_on='key', right_on=dfQuan_ms1.index) 
            
            avg_L_per_in_pep  = pd.DataFrame()
            max_dist_bet_mixed_heavy_scan = np.float(params["max_dist_bet_mixed_heavy_scan"])
            for pep in psm_K2_heavy.Peptide.unique():
                heavy_pep = (psm_K2_heavy[psm_K2_heavy.Peptide.str.contains(pep)]).drop_duplicates('key').reset_index(drop = True)
                mixed_pep = (psm_K2_mixed[psm_K2_mixed.Peptide.str.contains(pep)]).drop_duplicates('key').reset_index(drop = True)
                
                heavy_pep = heavy_pep[heavy_pep.frac.isin(mixed_pep.frac)]
                mixed_pep = mixed_pep[mixed_pep.frac.isin(heavy_pep.frac)]
                for frac_ in mixed_pep.frac.unique():
                    heavy_pep_ = (heavy_pep[heavy_pep.frac.str.contains(frac_)]).drop_duplicates('key').reset_index(drop = True)
                    MS1int_sum = heavy_pep_[['mixed_MS1_intensity', 'heavy_MS1_intensity']].sum(axis=1)
                    topIndex = MS1int_sum.sort_values(ascending=False).index[:1]
                    heavy_pep_ = heavy_pep_.loc[topIndex].reset_index(drop = True)
                    heavy_pep_[reporters] = heavy_pep_[reporters].replace(0,1)
                    
                    mixed_pep_ = (mixed_pep[mixed_pep.frac.str.contains(frac_)]).drop_duplicates('key').reset_index(drop = True)
                    MS1int_sum = mixed_pep_[['mixed_MS1_intensity', 'heavy_MS1_intensity']].sum(axis=1)
                    topIndex = MS1int_sum.sort_values(ascending=False).index[:1]
                    mixed_pep_ = mixed_pep_.loc[topIndex].reset_index(drop = True)
                    
                    heavy_and_mixed_psm = heavy_pep_[['mixed_MS1_intensity', 'heavy_MS1_intensity']].append(mixed_pep_[['mixed_MS1_intensity', 'heavy_MS1_intensity']]).reset_index(drop = True)
                    heavy_and_mixed_psm = heavy_and_mixed_psm[(heavy_and_mixed_psm > 0).all(1)]
                    psmSum = heavy_and_mixed_psm.sum(axis=1)
                    topIndex = psmSum.sort_values(ascending=False).index[:1]
                    heavy_and_mixed_MS1int = heavy_and_mixed_psm.loc[topIndex]
                    if (abs(np.int(mixed_pep_.key[0].split("_")[-1]) - np.int(heavy_pep_.key[0].split("_")[-1])) < max_dist_bet_mixed_heavy_scan) &  (len(heavy_and_mixed_MS1int) > 0):
                        mixed_pep_['ratio_ms1_ms2'] = (heavy_and_mixed_MS1int['mixed_MS1_intensity'].values/mixed_pep_[reporters].sum(axis=1)).values
                        mixed_pep_[reporters] = mixed_pep_.apply(lambda row: row['ratio_ms1_ms2']*row[reporters],axis=1)
                        heavy_pep_['ratio_ms1_ms2'] = (heavy_and_mixed_MS1int['heavy_MS1_intensity'].values/heavy_pep_[reporters].sum(axis=1)).values
                        heavy_pep_[reporters] = heavy_pep_.apply(lambda row: row['ratio_ms1_ms2']*row[reporters],axis=1)
            
                        mixed_pep_[reporters] = mixed_pep_[reporters].values/heavy_pep_[reporters].values 
                        mixed_pep_[reporters] = 2/(2+mixed_pep_[reporters].values) 
                        mixed_pep_[reporters] = 1-mixed_pep_[reporters].values 
                        mixed_pep_['scan_difference'] = abs(np.int(mixed_pep_.key[0].split("_")[-1]) - np.int(heavy_pep_.key[0].split("_")[-1]))
                        avg_L_per_in_pep = pd.concat([avg_L_per_in_pep, mixed_pep_],axis=0)
                        
            avg_L_per_in_pep[fully_light_reporters] = 1
            req_col = ['Peptide'] + reporters + ['mixed_MS1_intensity','heavy_MS1_intensity', 'scan_difference']
            outfile = open(os.path.join(saveDir_lys, 'free-Lys_percentage_in_diK_peptides.txt'),"w")
            avg_L_per_in_pep[req_col].to_csv(outfile, sep = "\t", line_terminator='\n', mode='a',index = None)
            outfile.close()
            trim_peptide_oulier_percent = np.float(params["trim_peptide_oulier_percent"])
            Q1 = avg_L_per_in_pep[reporters].quantile(trim_peptide_oulier_percent/100)
            Q3 = avg_L_per_in_pep[reporters].quantile(1 - (trim_peptide_oulier_percent/100))
            #df = avg_L_per_in_pep[~((avg_L_per_in_pep[reporters] < Q1) |(avg_L_per_in_pep[reporters] > Q3)).any(axis=1)]
        
            mask = (avg_L_per_in_pep[reporters] < Q1) | (avg_L_per_in_pep[reporters] > Q3)
            avg_L_per_in_pep_ = avg_L_per_in_pep[reporters]
            avg_L_per_in_pep_[mask] = np.nan
        
            median_light_free_K = pd.DataFrame(avg_L_per_in_pep_.median(0) )
            median_light_free_K.rename(columns={0:'median-light-L%'},inplace=True)
            median_light_free_K.index.name = 'reporters'
                                       
            outfile = open(os.path.join(saveDir_lys, 'median_free-Lys_percentage_in_diK_peptides.txt'),"w")
            median_light_free_K.to_csv(outfile, sep = "\t", line_terminator='\n', mode='a')#,index = None
            outfile.close()
        ####################
               
                    
