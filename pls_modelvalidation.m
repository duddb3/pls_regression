function mdl = pls_modelvalidation(X,Y,ncomp)
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
    parfor p=1:n_perms
        C = cvpartition(size(Y,1),'KFold',K);
    
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
            CoD_perm(p,k) = 1 - PRESS_perm(p,k)/TSS_perm(p,k);

            

        end
        CorrCoef(p,:) = diag(corr(Y,allyhat));
        CorrCoef_perm(p,:) = diag(corr(Y_perm,allyhat_perm));
    end
    TSS = mean(TSS,2);
    TSS_perm = mean(TSS_perm,2);
    PRESS = mean(PRESS,2);
    PRESS_perm = mean(PRESS_perm,2);
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
    mdl.CoD = CoD;
    mdl.CoD_perm = CoD_perm;
    % The model p-value is the fraction of CoD values from the null
    % distribution that are greater than the estimated real CoD value
    mdl.pvalue = (1 + sum(CoD_perm(:)>mean(CoD(:))))/(n_perms +1);

    % Model r and r-squared (1 value for each response variable)
    mdl.r = mean(CorrCoef);
    mdl.r_perm = CorrCoef_perm;
    mdl.rsquared = mdl.r.^2;
    
    % Now run the full model for weights and prediction
    [~,~,~,~,mdl.FullModelWeights] = plsregress(X,Y,ncomp);

end