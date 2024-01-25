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
    Q2 = TSS;
    Q2_perm = TSS;

    parfor p=1:n_perms
        C = cvpartition(size(Y,1),'KFold',K);
    
        % shuffle Y
        Y_perm = Y(randperm(size(Y,1),size(Y,1)),:);

        for k=1:K   % do each fold        
            % Get the training and test sets
            trn = training(C,k);
            tst = test(C,k);
        
            % Train the model with the kth partition data
            [~,~,~,~,BETA] = plsregress(X(trn,:),Y(trn,:),ncomp);
            % Fill in the predicted estimates for the kth test data
            yhat = [ones(sum(tst),1) X(tst,:)]*BETA;
            % Calculate the Predicted REsidual Sum of Squares
            PRESS(p,k) = sum((Y(tst,:)-yhat).^2,'all');
            % Calculate the total sum of squares
            TSS(p,k) = sum((Y(tst,:)-mean(Y(tst,:))).^2,'all');
            Q2(p,k) = 1 - PRESS(p,k)/TSS(p,k);

            % Train the model with the kth partition data for shuffled Y
            [~,~,~,~,BETA] = plsregress(X(trn,:),Y_perm(trn,:),ncomp);
            % Fill in the predicted estimates for the kth test data
            yhat_perm = [ones(sum(tst),1) X(tst,:)]*BETA;
            % Calculate the Predicted REsidual Sum of Squares
            PRESS_perm(p,k) = sum((Y_perm(tst,:)-yhat_perm).^2,'all');
            % Calculate the total sum of squares
            TSS_perm(p,k) = sum((Y_perm(tst,:)-mean(Y_perm(tst,:))).^2,'all');
            Q2_perm(p,k) = 1 - PRESS_perm(p,k)/TSS_perm(p,k);

            

        end
    end
    

    rng = linspace(...
        min([Q2;Q2_perm],[],'all'),...
        max([Q2;Q2_perm],[],'all'),...
        100);
    figure,histogram(Q2,rng);
    hold on
    histogram(Q2_perm,rng)
    
    hold off
    set(gcf,'Color','w')
    legend({'Model','Null Model'})
    xlabel('Predictive R^2')

    mdl.TSS = TSS;
    mdl.TSS_perm = TSS_perm;
    mdl.PRESS = PRESS;
    mdl.PRESS_perm = PRESS_perm;
    mdl.Q2 = Q2;
    mdl.Q2_perm = Q2_perm;
    % The model p-value is the fraction of Q^2 values from the null
    % distribution that are greater than the estimated real Q^2 value
    mdl.pvalue = sum(Q2_perm(:)>mean(Q2(:)))/n_perms;

end