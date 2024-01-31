# pls_regression
A set of MATLAB functions to perform a partial least squares (PLS) regression analysis. You need:
   - A set of predictor variables X. X is a matrix of size n-by-p where n is the number of observations and p is the number of predictor variables
   - A set of response variables Y. Y is a vector or matrix of size n-by-m where n is the number of observations and m is the number of response variables

The PLS regression analysis will tell you A) the predictive value of your matrix X for your response variable(s) Y and whether that value is significant and B) provide you with the coefficient estimates and statistical measure for each of your predictor variables (i.e., tell you which of your predictor variables are significant in predicting responses in Y). This is done in three steps:

1. **Find the optimal number of components**
   >ncomp = pls_optimalcomp(X,Y);  
   
   This function finds the optimal hyperparameter for the model: the number of components to retain. It performs 5-fold cross validation of the PLS model for each case of components from 1 to 30*; for each case, the average mean squared error and predictive **R<sup>2</sup>** is calculated from the predicted responses from the testing set. The optimal number of components is the case that minimizes mean squared error or maximizes predictive **R<sup>2</sup>**. Plots of these model performance metrics vs. the number of components are displayed (for fun). Note: in order to obtain more robust estimates of the model performance metrics, the process is repeated with 20 different holdout sets for cross validation.

   *Theoretically, you can have up to rank(X) number of components. However, because PLS components are derived by taking into account the response variable(s), the optimal number of components is typically fairly low.   
   
3. **Perform model validation**
   >mdl = pls_modelvalidation(X,Y,ncomp);
   
   This function performs model validation via permutation. Specifically, for each permutation a new cross-validation partition set is defined and a random permutation of the response variable(s) is generated. Then for each fold, a partial least squares regression model is fit to the training set; the resulting model coefficients are used to generate the predicted response variables yhat (for the regular data) and yhat_perm (for the permuted data). The mean squared error, Predicted REsidual Sum of Squares, Total Sum of Squares, and predictive **R<sup>2</sup>** are then calculated for each permutation. The resulting distributions are compared to yield the overall model significance.
   
5. **Obtain coefficients and statistics**
   >[coeff,zstats,pvals] = pls_modelweights(X,Y,ncomp);


# Notes for using with neuroimaging data loaded by Canlab:
Say you created an fmri object data structure using the Canlab tools like so:  
>fmri_obj = fmri_data(filelist,mask);  
>fmri_obj.Y = your_response_variable;

Then you can use the following commands to run the pls regression analysis:  
>X = fmri_obj.dat';  
>Y = fmri_obj.Y;  
>ncomp = pls_optimalcomp(X,Y);  
>mdl = pls_modelvalidation(X,Y,ncomp);  
>[coeff,zstats,pvals] = pls_modelweights(X,Y,ncomp);  

Then you can use the following commands to put your coefficients and p-values into an object to use Canlabs visualization tools:
>stat_obj = statistic_image;  
>stat_obj = fmri_obj.volInfo;  
>stat_obj.dat = coeff';  
>stat_obj.p = pvals';

Then you just make calls to whatever Canlab visualization tool you want, e.g.:
>orthviews(stat_obj);
