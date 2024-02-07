# JUMPsilactmt-Version-1.0.0
## Contents <br>
•	Introduction <br>
•	Release notes <br>
•	Software and Hardware Requirements <br>
•	Installation <br>
•	Input Files and their Preparation <br>
•	Update the parameter file <br>
•	Run the JUMPsilactmt program <br> 
•	Output files <br> 
•	Sample program for joining protein quantification data from two batches and performing a QC  and its sample parameter file <br>
•	Maintainers <br>
•	Acknowledgments <br>
•	References <br>

## Introduction <br>
JUMPt (JUMP-turnover) software determines the protein turnover rates in pulse SILAC-labeled animals using mass spectrometry (MS) data. JUMPt uses a novel differential equation-based mathematical model to calculate reliable and accurate protein turnover rates. The proposed method calculates individual proteins' half-lives (Corrected half-life) by fitting the dynamic data of unlabeled free Lys and protein-bound Lys from individual proteins. Besides, the program calculates proteins' apparent half-lives also using exponential functions.
JUMPt is part of JUMP Software Suite (shortly JUMP), is an ongoing large software suite developed for the need of mass spectrometry (MS)- based proteomics, metabolomics, and the integration with genomics for network analysis at the level of systems biology. Currently, JUMP can handle protein/peptide database creation, database search, identification filtering, quantification, network, proteogenomic, and protein turnover analysis.

## Release notes (Version 2.0.0) <br>
In the current version 
1. We assume that the mice's overall amount of proteins is unchanged over time as the mice used in this study are adults. 
2. The Lys used in protein synthesis originates from the food intake, with the rest recycled through protein degradation. 
3. Concentration of soluble Lys is conserved; the rate of free Lys absorbed from food is assumed to equal the rate excreted. 
4. Three different settings, which represent simple to comprehensive inputs. Here, setting 2 is the default in the current version.
5. Here, the default optimization algorithm is nonlinear least-squares.
5. The current version supports different time point inputs, such as 3, 4, and 5 point inputs, including 0 days.
6. This version can also calculate apparent half-lives.

## Software and Hardware Requirements <br>
The program was written in MATLAB language. The program runs on any Linux, Mac, or Windows computer with MATLAB R2014 (The MathWorks, Inc., Natick, Massachusetts, United States) or the above version. The current JUMPt program has been successfully tested with MATLAB R2021 version on the system: 16 GB memory and 3.3 GHz CPU processors with six cores. The program needs more time to complete on the system with fewer core processors in the CPU.
MATLAB toolbox needed: 
- Global Optimization toolbox along with other basic toolboxes

## Installation <br>
Installation of the script is not required. Download all the files/folders to any working directory (e.g., /home/usr/JUMPt) as shown in Figure 1. 

![Figure1](https://github.com/abhijitju06/JUMPt/assets/34911992/d5ff202e-c1a7-4d23-bb80-151f26b1028b)
<p align="center">
Figure 1
</p>

## Input File Preparation <br>
A testing dataset with 100 proteins is available for each setting. Besides, the testing dataset for different time points is also available for default setting 2. Like the testing dataset, the user must prepare the input data file with the information below.
•	pSILAC proteins (mandatory)
•	pSILAC free (unbound) Lys (optional)
•	Free Lys concentration (required, as setting 2 is the default in the current version)
•	Lys concentration in individual proteins (optional)

## Update the parameter file <br>
The JUMPt program requires a parameter file (JUMPt.parms). The user must specify the following parameters in the 'JUMPt.params' file.
•	JUMPt setting 
•	Input file name (along with the exact path)
•	Bin size('bin_size') to fit proteins each time 
•	MATLAB optimization algorithm
•	Number of time points
•	Purity of SILAC food 
•	Whether the user wants to calculate the apparent half-life

## Run the JUMPt program (Demo data set) <br>
Launch the MATLAB software and open the JUMPt main program file "Run_Main_File.m" in it, as shown in Figure 2. Press the "Run' button, as shown in the figure, to start the program. Once the program begins, it will show the progress of protein fitting and the successful completion (Figure 3).

![Figure2](https://github.com/abhijitju06/JUMPt/assets/34911992/8a4aeaf1-d008-4a23-a0e6-6ef08033c1ca)
<p align="center">
Figure 2
</p>

Nonlinear fitting of proteins and Lys data using ODE is computationally expensive, especially when the protein data is enormous (e.g.,> 1000 proteins). We divide the proteins into sets with bin sizes between 10-100 to reduce the computational complexity. The program finds the optimal degradation rates (turnover rates or half-lives) by fitting protein data (in setting-1) and free-Lys data (in setting-2 and setting-3).

![Figure3](https://github.com/abhijitju06/JUMPt/assets/34911992/d2bf25a5-32e6-43e0-a8fd-8117dd000fa0)
<p align="center">
Figure 3
</p>

## Output file information <br>
Two output Excel files are generated with the prefix 'results_Corrected_T50' and 'results_Apparent_T50' to the input file name. They will be rendered in the same folder where the input file is located. The results with proteins' corrected half-lives (in days) and confidence intervals will be saved to the output file. In addition, parameters used to calculate the half-lives will also be kept in the output file. Besides, the program will save the apparent half-lives in the output file.

## Maintainers <br>
To submit bug reports and feature suggestions, please contact:
Surendhar Reddy Chepyala (surendharreddy.chepyala@stjude.org), Junmin Peng (junmin.peng@stjude.org), Abhijit Dasgupta (abhijit.dasgupta@stjude.org), and Jay Yarbro (jay.yarbro@stjude.org). 

## Acknowledgment <br>
We acknowledge St. Jude Children's Research Hospital, ALSAC (American Lebanese Syrian Associated Charities), and the National Institute of Health for supporting the development of the JUMP Software Suite.

## References <br>
1. Chepyala et al., JUMPt: Comprehensive protein turnover modeling of in vivo pulse SILAC data by ordinary differential equations. Analytical chemistry (under review)
2. Wang, X., et al., JUMP: a tag-based database search tool for peptide identification with high sensitivity and accuracy. Molecular & Cellular Proteomics, 2014. 13(12): p. 3663-3673.
3. Wang, X., et al., JUMPm: A Tool for Large-Scale Identification of Metabolites in Untargeted Metabolomics. Metabolites, 2020. 10(5): p. 190.
4. Li, Y., et al., JUMPg: an integrative proteogenomics pipeline identifying unannotated proteins in human brain and cancer cells. Journal of proteome research, 2016. 15(7): p. 2309-2320.
5. Tan, H., et al., Integrative proteomics and phosphoproteomics profiling reveals dynamic signaling networks and bioenergetics pathways underlying T cell activation. Immunity, 2017. 46(3): p. 488-503.
6. Peng, J., et al., Evaluation of multidimensional chromatography coupled with tandem mass spectrometry (LC/LC-MS/MS) for large-scale protein analysis: the yeast proteome. Journal of proteome research, 2003. 2(1): p. 43-50.
7. Niu, M., et al., Extensive peptide fractionation and y 1 ion-based interference detection method for enabling accurate quantification by isobaric labeling and mass spectrometry. Analytical chemistry, 2017. 89(5): p. 2956-2963.
