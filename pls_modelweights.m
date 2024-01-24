function [b_mean_coeff,b_Z_coeff,b_P_coeff] = pls_modelweights(X,Y,ncomp)
    % This function performs bootstrapping of partial least squares
    % regression with inputs:
    %
    %   X: an n-by-v matrix of predictors where n is the number of 
    %       observations (e.g., participants) and p is the number of
    %       predictors.
    %   Y: an n-by-m matrix of response variables where n is the number of
    %       observations (e.g., participants) and m is the number of
    %       response variables being predicted
    %   ncomp: a scalar number that is the number of components to use in
    %       the partial least squares regression
    %
    % The outputs of the function are:
    %   b_mean_coeff: a v-by-m matrix of coefficient estimates for the
    %       partial least squares regression
    %   b_Z_coeff: a corresponding v-by-m matrix of z-statstics (i.e., how 
    %       significantly each predictor variable contributes to the model
    %       fit of each response variable)
    %   b_P_coeff: the corresponding p-values (two-tailed)


    nboots = 5000;
    bs_b = bootstrp(nboots,@bootplsNCdim,X,Y,ncomp,'options',statset('UseParallel',true));
    r_bs_b = reshape(bs_b,nboots,1+size(X,2),[]);
    r_bs_b = r_bs_b(:,2:end,:);

    b_mean_coeff = squeeze(mean(r_bs_b,'omitnan'));
    b_ste_coeff = squeeze(std(r_bs_b,'omitnan'));
    b_ste_coeff(b_ste_coeff==0) = Inf;
    b_Z_coeff = b_mean_coeff ./ b_ste_coeff;
    b_P_coeff = 2*normcdf(-1*abs(b_Z_coeff),0,1);


    function BETA = bootplsNCdim(x,y,nc)
        [~,~,~,~,BETA] = plsregress(x,y,nc);
    end

end