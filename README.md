# JUMPsilactmt-Version-1.0.0
## Contents <br>
•	Introduction <br>
•	Release notes (Version 1.0.0) <br>
•	Software and Hardware Requirements <br>
•	Installation <br>
•	Input Files and their Preparation <br>
•	Update the parameter file <br>
•	Run the JUMPsilactmt program <br> 
•	Output files <br> 
•	Maintainers <br>
•	Acknowledgments <br>
•	References <br>

## Introduction <br>
<div align="justify"> 
The JUMP (Jumbo Mass Spectrometry-based Proteomics) search engine revolutionized peptide and protein identification by amalgamating pattern-based scoring and de novo tag scoring, significantly elevating the precision of peptide-spectrum matches (PSMs), as showcased in the study by Wang et al. (2014). Besides, the JUMP filtering engine conducted a comprehensive analysis of database search results, aiming to enhance the precision of peptide and protein identification, with insights from the works of Peng et al. (2003) and Xu et al. (2009). In the context of multi-batch TMT analysis, the JUMP batch program effectively addressed challenges related to systematic non-biological batch variations and the exclusive identification of specific peptides and proteins in certain batches.<br><br>
JUMPsilactmt integrated data inputs from the JUMP batch program and the mzXML files of individual samples within each batch. It encompassed a two-fold quantification approach involving MS1 level quantification/normalization (specifically SILAC quantification) and MS2 level quantification. The workflow started with extracting TMT reporter ion peaks and consolidating m/z shifts among these reporters. Furthermore, it utilized an additional round of TMT reporter ion peak extraction in the initial phases of MS2 level quantification. The process assigned the lowest intensity across the entire batch to the respective PSMs if the reporters lacked quantification.<br><br> 
 Additionally, the process applied a TMT impurity correction, assigning a value of 1 to reporters with an intensity of <=0 and subsequently incorporating the relevant proteins and peptides into the PSMs. Notably, this TMT impurity correction deviated slightly from the standard impurity correction used in our JUMP-q pipeline (Niu et al., 2017). In this case, the approach shifted from the previous evaluation of the final PSM quantification based on a maximum of (original reporter intensity/2) and impurity-corrected intensity to utilizing the impurity-corrected reporter intensities. The subsequent stages of MS2-level quantification included the application of loading bias and TMT noise corrections. Following the approach of Niu et al. (2017), the loading bias correction aimed to eliminate systematic sample processing discrepancies by normalizing reporter intensities for individual Lys PSMs. It involved the division of these intensities by PSM-wise mean values and their subsequent conversion into log-scale values. <br><br>
Consequently, the majority of PSM intensities centered around zero across the reporters. The trimmed mean intensity for each reporter was generated by excluding the top and bottom 10% of values, providing the necessary normalization factors that were then converted to raw-scale single Lys PSMs. The process excluded PSMs containing decoys, contaminants, or anomalies.<br><br>
The TMT noise correction process commenced with determining the minimum intensity observed across the entire batch. It is initiated by identifying noise channels for both light PSMs (representing the fully labeled channel) and heavy PSMs (representing the fully unlabeled channel). Subsequently, it set the TMT noise level for each PSM based on the first and second lowest signals. After this determination, it subjected both heavy and light PSMs to noise correction at varying levels as dictated by the TMT noise. Each light and heavy PSM underwent normalization based on the respective total intensity, and heavy PSMs were aggregated at the peptide and protein levels. The normalization involved the MS1 intensity ratio. This method encompassed the calculation of heavy and light peptide isotopic distributions for all identified peptides, including extracting heavy and light peptide MS1 intensities from each scan, focusing on the most significant peaks. It computed heavy and light peptide MS1 precursor intensities ratios for each PSM. Subsequently, it summarized the MS1 intensity ratio at the peptide level and normalized TMT noise-corrected light PSMs using the corresponding MS1 L/H ratio. After summarizing TMT noise-corrected light and heavy PSMs into the related peptides, it calculated each peptide's light-Lys% (L/(L+H)) using heavy and light peptides. Finally, it identified and removed outliers using the extreme studentized deviation (ESD) and Dixson Q-test methods. The entire pipeline concluded with the ultimate protein summarization.<br><br>
</div>
 
<div align="justify"> 
In addition to protein quantification, JUMPsilactmt was vital when an experimental measurement of the free lysine pSILAC ratio was unavailable. However, among all identified peptides, tiny portions contained two Lys residues (i.e., double-K-peptides).  The double-K-peptides should have three peaks (light, mixed, and heavy). The light peak may also be generated from pre-existing light proteins. <br><br>
</div>

If each protein ($P$) had a synthesis rate ($S$) following zero-order kinetics, and $H_A$% was the averaged percentage of heavy K, considering time $t$, we can derive the following: <br>

 
Total $P_M$ (mixed) $=$ $S \times t  \times$  $H_A$% $\times$ $(1-H_A$% $) \times 2$ (as both Lys had an equal chance to be heavy) <br>
Total $P_H$ (heavy) $= S \times t \times H_A$% $\times$ $H_A$% <br>
The ratio of the mixed to the heavy peptide ${ R (=  \frac{P_M}{P_H}  ) =   \frac{(1-H_A) \times 2}{H_A}}$ was independent of synthesis rates. <br>
Thus, <br>
                                         $H_A$% $= \frac{2}{(2+R)}$    <br>                             
                                         $L_A$% $= 1-H_A$% <br>
We can use the above simple equations to derive the averaged $L_A$% during the pulse (e.g., eight days) based on double-K-peptides. <br>
As the $L$% is dynamic, the $L$% on day 8 was not equal to the averaged $L_A$% in 8 days. We should be able to fit the averaged $L_A$% in 8 days to derive $L$% on day 8. <bR> 

<div align="justify"> 
The free lysine estimation process was like the quantification process. Unlike the quantification process, we derived double Lys PSM during the loading bias correction instead of single Lys PSM. Moreover, we inspected mixed and heavy PSMs in the TMT noise correction step and summarized them only to peptide level. The MS1 level quantification to peptide level and the mathematical procedure helped get the free lysine pSILAC ratio. 
</div>



## Release notes (Version 1.0.0) <br>
<div align="justify"> 
In the current version <br> 
1. We can quantify proteins from multiple batches of samples generated by the SILAC-TMT-LC/LC-MS/MS process. <br>
2. We can also estimate free lysine decay from this data if experimental free lysine information is unavailable. 
</div>

## Software and Hardware Requirements <br>
<div align="justify"> 
The program was written in Python (version 3.8.1) language. The program runs on any Linux environment. The current JUMPsilactmt program has been successfully tested on the High-Performance Computing Facility (HPCF) of St. Jude Children's Research Hospital, Memphis, USA. Thus, the program requires at least a system with  64 GB memory and 3.3 GHz CPU processors with sixteen cores. The program needs more time to complete on the system with fewer core processors in the CPU. <br>
• It needs the JUMP module to be loaded.
</div>

## Installation <br>
<div align="justify"> 
Installation of the script is not required. Download all the files/folders to any working directory (e.g., /home/usr/). 
</div>



## Input Files and their Preparation <br>

<div align="justify"> 
As shown in Figure 1, after metabolic labeling, tissue collection, and sample preparation, the LC/LC-MS/MS procedure generates the raw file, which is further converted to an mzXML file through a software tool, for example, msConvert. Then, one should run JUMP-search to identify heavy and light peptides from the SILAC database. The user should keep the search results in the working directory and filter them through JUMP-filter, followed by JUMP-batch-d programs. The search results and batch_id_results (kept in the same working directory) were the inputs of the JUMPsilactmt program. The user can download the sample input files from the link below.
</div>

https://drive.google.com/drive/folders/1VRKWWJVgPSKCH_HU1_tiYiVo_ZdGxiD6?usp=sharing

![image](https://github.com/abhijitju06/JUMPsilactmt-Version-1.0.0/assets/34911992/8d2d45cd-75c3-4178-a86a-9ae41cf7e6d1)
<p align="center">
Figure 1
</p>


## Update the parameter file <br>
<div align="justify"> 
The JUMPsilactmt program requires multiple parameter files depending on the number of batches. We include some of the essential parameters below. The user can check the sample parameter files ("prot_quan_lysine_est_batch1.params" and "prot_quan_lysine_est_batch2.params" for samples from two batches) for other parameters and change accordingly. <br>
•	idtxt; This is the main output from JUMP-batch-id program <br>
•	id_res_folder; This is an associated folder generated by JUMP-batch-id program <br>
•	output_folder_protein; The output folder where protein quantification results will be stored <br>
•	output_folder_lysine; The output folder where free lysine estimation results will be stored <br>
•	tmt_reporters_used; TMT reporter information <br>
•	Sample information in for each sets of pulse experiment; for example set1, set2, and set3 in the sample parameter files <br>
•	noise_removal_in_lightPSM; Noise correction by fully heavy reporters ; 1 = Yes; 0 = No  <br> 
•	noise_removal_in_heavyPSM; Noise correction by fully light reporters ; 1 = Yes; 0 = No  <br> 
•	nc_level; Noise correction level; may vary from 0.1 to 0.95; It prevents overcorrection <br>
•	normalization; 1 = Normalize with MS1 intensity ratio (default); 2 = Normalize each channel in every PSM obtained using protein amount obtained by non-Lys PSMs  <br> 
•	free_lysine_estimation; 1= yes; 0= no <br>
•	ms1_peak_extraction_method; 1 = strongest intensity; 2 = closest to expected peptide mass; only if multiple peaks detected within mass tolerance  <br>
•	strong_isotopic_peaks; 1 = select strongest isotopic peaks within mass_tolerance  instead of merging isotopic peaks based on a weighted average of intensity; = 0 otherwise
</div>


## Run the JUMPsilactmt program <br>
<div align="justify"> 
Here are the instructions to run the program. <br>
Step 1: Copy all the files to your working folder on HPC. <br>
Step-2: Update the '.params' file with input file names and other parameters as necessary.<br>
Step-3: bsub -R "rusage[mem=15000]" -q standard -P Proteomics -Is bash (Optional and it is system specific to assign more memory into your workspace) <br>
Step-4: module load jump <br>
Step-5: python JUMPsilactmt.py Batch1.params Batch2.params ... BatchN.params <br>
• For example: If you have the sample parameter files, the command is: python JUMPsilactmt.py prot_quan_lysine_est_batch1.params prot_quan_lysine_est_batch2.params <br><br>
Here, we include a sample program "Sample_combine_two_batches_prot_quan_data_with_qc.py" with its parameter file "Sample_prot_quan_two_batch_join_qc.params" to show how a user can join the protein quantification results from two batches and perform a quality control depending on free Lys outlier (proteins having faster degradation curve than free lysine) and degradation pattern outlier (proteins not showing continuous degradation pattern). To run this program, the user should execute the following command:<br>
 • python Sample_combine_two_batches_prot_quan_data_with_qc.py Sample_prot_quan_two_batch_join_qc.params <br>
However, for multiple batch (>2) samples, the user should modify the program and its parameter file accordingly.
</div>



## Output files <br>
<div align="justify"> 
From the following link, the user can download the sample output files generated by the run of JUMPsilactmt and the sample program for joining protein quantification data from two batches and performing a QC using the sample input files (provided with the input file link).
</div>

https://drive.google.com/drive/folders/1X-282W2PrTh51saSF509AuAz66hjpvjK?usp=sharing

## Maintainers <br>
To submit bug reports and feature suggestions, please contact:
Surendhar Reddy Chepyala (surendharreddy.chepyala@stjude.org), Junmin Peng (junmin.peng@stjude.org), Abhijit Dasgupta (abhijit.dasgupta@stjude.org), Jay Yarbro (jay.yarbro@stjude.org), Xusheng Wang (xusheng.wang@stjude.org). 

## Acknowledgment <br>
<div align="justify"> 
We acknowledge St. Jude Children's Research Hospital, ALSAC (American Lebanese Syrian Associated Charities), and the National Institute of Health for supporting the development of the JUMP Software Suite.
</div>

## References <br>
<div align="justify"> 
1. Chepyala et al., JUMPt: Comprehensive protein turnover modeling of in vivo pulse SILAC data by ordinary differential equations. Analytical chemistry, 2021. 93(40): p. 13495-13504. <br>
2. Wang, X., et al., JUMP: a tag-based database search tool for peptide identification with high sensitivity and accuracy. Molecular & Cellular Proteomics, 2014. 13(12): p. 3663-3673. <br>
3. Wang, X., et al., JUMPm: A Tool for Large-Scale Identification of Metabolites in Untargeted Metabolomics. Metabolites, 2020. 10(5): p. 190. <br>
4. Li, Y., et al., JUMPg: an integrative proteogenomics pipeline identifying unannotated proteins in human brain and cancer cells. Journal of proteome research, 2016. 15(7): p. 2309-2320. <br>
5. Tan, H., et al., Integrative proteomics and phosphoproteomics profiling reveals dynamic signaling networks and bioenergetics pathways underlying T cell activation. Immunity, 2017. 46(3): p. 488-503. <br>
6. Peng, J., et al., Evaluation of multidimensional chromatography coupled with tandem mass spectrometry (LC/LC-MS/MS) for large-scale protein analysis: the yeast proteome. Journal of proteome research, 2003. 2(1): p. 43-50. <br>
7. Niu, M., et al., Extensive peptide fractionation and y 1 ion-based interference detection method for enabling accurate quantification by isobaric labeling and mass spectrometry. Analytical chemistry, 2017. 89(5): p. 2956-2963. <br>
8. Xu, P., et al., Systematical optimization of reverse-phase chromatography for shotgun proteomics. Journal of proteome research, 2009. 8(8), p. 3944-3950.
</div>
