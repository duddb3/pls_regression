function mdl = pls_modelvalidation(X,Y,ncomp,stratcol)
    % This function performs model validation via _permutation.
    % Specifically, for each _permutation a new cross-validation partition
    % set is defined and a random _permutation of the response variables is
    % generated. Then for each fold, a partial least squares regression
    % model is fit to the training set; the resulting model coefficients
    % are used to generate the predicted response variables yhat (for the
    % regular data) and yhat_perm (for the _permuted data). The mean squared
    % error, Predicted REsidual Sum of Squares, and Total Sum of Squares,
    % and predictive relevance of the model (Q^2) are then calculated for
    % each _permutation. The resulting distributions are compared to yield
    % the overall model significance.

    if exist("stratcol","var")
        % Create partition for stratified K-fold cross-validation
        grp = categorical(Y(:,stratcol));
    else
        grp = size(Y,1);
    end

    % Set number of _permutations to perform
    n_perms = 5000;

    % Set the number of folds for cross validation
    K = 5;

    

    % instantiate variables
    TSS = NaN(n_perms,K);
    TSS_perm = TSS;
    PRESS = TSS;
    PRESS_perm = TSS;
    CoD = TSS;
    CoD_perm = TSS;
    CorrCoef = NaN(n_perms,size(Y,2));
    CorrCoef_perm = CorrCoef;
    MAE = CorrCoef;
    MAE_perm = CorrCoef;
    MSE = CorrCoef;
    MSE_perm = CorrCoef;
    RMSE = CorrCoef;
    RMSE_perm = CorrCoef;
    parfor p=1:n_perms
        C = cvpartition(grp,'KFold',K);

        % shuffle Y
        Y_perm = Y(randperm(size(Y,1),size(Y,1)),:);

        allyhat = zeros(size(Y));
        allyhat_perm = allyhat;

        for k=1:K   % do each fold        
            % Get the training and test sets
            trn = training(C,k);
            tst = test(C,k);
        
            % Train the model with the kth partition data
            [~,~,~,~,BETA] = plsregress(X(trn,:),Y(trn,:),ncomp);
            % Fill in the predicted estimates for the kth test data
            yhat = [ones(sum(tst),1) X(tst,:)]*BETA;
            allyhat(tst,:) = yhat;
            % Calculate the Predicted REsidual Sum of Squares
            PRESS(p,k) = sum((Y(tst,:)-yhat).^2,'all');
            % Calculate the total sum of squares
            TSS(p,k) = sum((Y(tst,:)-mean(Y(tst,:))).^2,'all');
            % Calculate the Coefficient of Determination
            CoD(p,k) = 1 - PRESS(p,k)/TSS(p,k);

            % Train the model with the kth partition data for shuffled Y
            [~,~,~,~,BETA] = plsregress(X(trn,:),Y_perm(trn,:),ncomp);
            % Fill in the predicted estimates for the kth test data
            yhat_perm = [ones(sum(tst),1) X(tst,:)]*BETA;
            allyhat_perm(tst,:) = yhat_perm;
            % Calculate the Predicted REsidual Sum of Squares
            PRESS_perm(p,k) = sum((Y_perm(tst,:)-yhat_perm).^2,'all');
            % Calculate the total sum of squares
            TSS_perm(p,k) = sum((Y_perm(tst,:)-mean(Y_perm(tst,:))).^2,'all');
            % Calculate the Coefficient of Determination
            CoD_perm(p,k) = 1 - PRESS_perm(p,k)/TSS_perm(p,k);

            

        end
        CorrCoef(p,:) = diag(corr(Y,allyhat));
        CorrCoef_perm(p,:) = diag(corr(Y_perm,allyhat_perm));

        % Prediction error
        pred_err = Y-allyhat;
        pred_err_perm = Y_perm-allyhat_perm;
        % Mean absolute error
        MAE(p,:) = mean(abs(pred_err));
        MAE_perm(p,:) = mean(abs(pred_err_perm));
        % Mean squared error
        MSE(p,:) = mean(pred_err.^2);
        MSE_perm(p,:) = mean(pred_err_perm.^2);
        % Root mean squared error
        RMSE(p,:) = (mean(pred_err.^2)).^0.5
        RMSE_perm(p,:) = (mean(pred_err_perm.^2)).^0.5
    end
    CoD = mean(CoD,2);
    CoD_perm = mean(CoD_perm,2);
    rng = linspace(min([CoD;CoD_perm]),max([CoD;CoD_perm]),100);
    figure,histogram(CoD,rng);
    hold on
    histogram(CoD_perm,rng)
    hold off
    set(gcf,'Color','w')
    legend({'Model','Null Model'})
    xlabel('Model Coefficient of Determination')

    % Model coefficient of determination (1 value for whole model)
    mdl.CV.CoD = mean(CoD(:));
    mdl.CV.CoD_dist = CoD;
    mdl.CV.CoD_perm = CoD_perm;
    % The model p-value is the fraction of CoD values from the null
    % distribution that are greater than the estimated real CoD value
    mdl.CV.pvalue = (1 + sum(CoD_perm(:)>mean(CoD(:))))/(n_perms +1);

    % Model r and r-squared (1 value for each response variable)
    mdl.CV.r = mean(CorrCoef);
    mdl.CV.r_dist = CorrCoef;
    mdl.CV.r_perm = CorrCoef_perm;
    mdl.CV.rsquared = mdl.CV.r.^2;

    % Mean squared error, root mean squared error, and mean absolute error
    mdl.CV.MSE = mean(MSE);
    mdl.CV.MSE_dist = MSE;
    mdl.CV.MSE_perm = MSE_perm;
    mdl.CV.RMSE = mean(RMSE);
    mdl.CV.RMSE_dist = RMSE;
    mdl.CV.RMSE_perm = RMSE_perm;
    mdl.CV.MAE = mean(MAE);
    mdl.CV.MAE_dist = MAE;
    mdl.CV.MAE_perm = MAE_perm;
    
    % Now run the full model for weights and prediction
    [mdl.FullModel.xl,mdl.FullModel.yl,...    
        mdl.FullModel.xs,mdl.FullModel.ys,...
        mdl.FullModel.Weights,mdl.FullModel.pctvar] = plsregress(X,Y,ncomp);
    mdl.FullModel.Prediction = [ones(size(X,1),1) X]*mdl.FullModel.Weights;

    % Now run model permutations to test significance for loadings amd
    % weights
    yl_perm = nan([size(mdl.FullModel.yl) n_perms]);
    xl_perm = nan([size(mdl.FullModel.xl) n_perms]);
    beta_perm = nan([size(mdl.FullModel.Weights) n_perms]);
    pctvarp = nan([size(mdl.FullModel.pctvar) n_perms]);
    parfor p=1:n_perms
        Yperm = Y(randperm(size(Y,1),size(Y,1)),:);
        [xl_perm(:,:,p),yl_perm(:,:,p),~,~,beta_perm(:,:,p),pctvarp(:,:,p)] = plsregress(X,Yperm,ncomp);
    end
    mdl.FullModel.PctVar_pvalue = (sum(pctvarp>mdl.FullModel.pctvar,3)+1)/(n_perms+1);
    mdl.FullModel.yl_pvalue = (sum(abs(yl_perm)>abs(mdl.FullModel.yl),3)+1)/(n_perms+1);
    mdl.FullModel.yl_qvalue = reshape(mafdr(mdl.FullModel.yl_pvalue(:),'BHFDR',true),size(mdl.FullModel.yl_pvalue));
    mdl.FullModel.xl_pvalue = (sum(abs(xl_perm)>abs(mdl.FullModel.xl),3)+1)/(n_perms+1);
    for n=1:ncomp
        mdl.FullModel.xl_qvalue(:,n) = mafdr(mdl.FullModel.xl_pvalue(:,n),'BHFDR',true);
    end
    mdl.FullModel.Weights_pvalue = (1 + sum(abs(beta_perm)>abs(mdl.FullModel.Weights),3))/(n_perms+1);



end