Datasets
--------------------
Both simulation datasets are included in the Data.zip folder. Predictor values are provided in the CSV files, while the true coefficients are stored as attributes in the accompanying shapefiles. Note that all data were standardized during the data-generating process.

Citation
--------------------
This repository belongs to the M-SGWR local regression model, and the full preprint article can be found through this DOI:  https://doi.org/10.48550/arXiv.2601.19888.
For citation, please cite the most updated version of the article.

M-SGWR Implementaion
---------------------
In the current version, you first need to copy and paste all these python files in your jupyter notebook or any other environment that you work. Then use the 'call_function.py' for executing the model but make sure you have other python files in your environment first in this order: 
- (1) BW_alpha_optimization.py
- (2) Diagnostics.py
- (3) bw_optimization.py, and
- (4) main_class.py.  

Parameter Extraction 
-----------------------------------------------------
- msgwr_res.R2
- msgwr_res.adj_R2
- msgwr_res.aicc
- msgwr_res.aic
- msgwr_res.params
- msgwr_res.bse
- msgwr_res.localR2
- msgwr_res.filter_tvals()
- msgwr_res.filter_tvals(alpha=0.05) >>> t values with 95% confidence interval
- msgwr_res.summary()
  
Data Format 
-------------
Input data must be a CSV file with the following column order:
- longitude, latitude, dependent_variable, independent_variable_1, ..., independent_variable_n
- In the attached figure, "y" stands for dependent variable, "x1,x2,...xn" stand for independent variables
- "Longitude" and "Latitude" are the coordinate

![data format](https://github.com/user-attachments/assets/e5e6547d-5eb0-444a-a9be-8b315cbf9997)

Categorical Variables
---------------------
Categorical variables must be pre-processed into dummy variables.
Example: For a 3-class variable ("urban", "peri-urban", "rural"), create:
- urban_dummy: 1 if urban, else 0
- peri_urban_dummy: 1 if peri-urban, else 0
- Rural becomes the reference class (excluded)

![Example](https://github.com/user-attachments/assets/08a252df-c9ef-414a-ba30-a41914016e50)

Model Overview
------------
M-SGWR (Multiscale Similarity and Geographically Weighted Regression) is a multiscale local regression model designed to capture complex spatial heterogeneity by jointly accounting for geographic proximity and attribute similarity at the predictor level.

Traditional local regression models such as GWR and MGWR assume that spatial variation in relationships is governed solely by geographic distance. While MGWR allows each predictor to operate at a different spatial scale, it still enforces spatial smoothness through distance-based kernels alone. As a result, these models may struggle to recover localized or fragmented coefficient patterns that are not purely driven by geography.

M-SGWR extends the SGWR framework by introducing a predictor-specific mixing parameter (𝛼) that controls the relative contribution of geographic proximity and attribute similarity in the weighting process. This multiscale formulation enables M-SGWR to explicitly identify whether each predictor exhibits:
- Purely geographic effects (distance-driven),
- Purely attribute-driven effects (similarity-driven), or
- Mixed effects, arising from both spatial proximity and attribute similarity.

Unlike the original SGWR model (https://doi.org/10.1080/13658816.2024.2342319), which applies a single global mixing parameter and cannot distinguish mixed behavior across predictors, M-SGWR provides variable-specific inference on the nature of spatial heterogeneity. This allows researchers to determine which predictors vary smoothly over space and which exhibit localized, non-smooth variation tied to attribute regimes.

Overall, M-SGWR offers a flexible and interpretable framework for modeling complex spatial processes, making it particularly well suited for applications where spatial relationships are influenced by both geography and contextual similarity, such as social, environmental, and public health studies. Note: When all predictors exhibit purely geographic behavior (i.e., their optimal mixing parameters converge to one), the M-SGWR model naturally reduces to the MGWR model. In this case, attribute similarity has no influence on the weighting process, and M-SGWR becomes equivalent to MGWR without requiring any additional assumptions or constraints.

![comparison](https://github.com/user-attachments/assets/273e78a2-b9a1-48ad-83bd-4168412822d2)

![correlationi](https://github.com/user-attachments/assets/44ab2c8a-6db5-4a81-9c93-c2cfc8803a18)

