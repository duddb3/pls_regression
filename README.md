# pls_regression
A set of MATLAB functions to perform a partial least squares regression analysis.


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
