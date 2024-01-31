function ncomp = pls_optimalcomp(X,Y)
    % This function finds the optimal number of components to perform
    % partial least squares regression for predictor variabls X and
    % response variables Y.

    % Number of folds for cross-validation
    K = 5; 
    % Number of permutations (generating different cross-validation sets)
    P = 20; 
    % Choose upper bound on number of components to test. You could go up 
    % to rank(X), but that is very high
    mxncomp = min(30,rank(X));
    % Instantiate array that we will fill with the predictive relevance of 
    % each fold at each dimensionality
    Q2 = NaN(P,K,mxncomp);
    MSE = NaN(P,K,mxncomp);
    
    
    
    parfor p=1:P
        % Create cross-validation partitions for K-folds
        n_subjects = size(Y,1);
        C = cvpartition(n_subjects, 'KFold', K);
        for c=1:mxncomp   % for each level of dimensionality
            for k=1:K   % for each fold
                % Get train and test set indices for fold k
                traini = training(C,k);
                testi = test(C,k);
    
                % fit the PLSR model with ncomp PLS-scores on training fold
                [~,~,~,~,b_pls,~] = plsregress(X(traini,:),Y(traini,:),c);
                % Evaluate fit on test fold
                yhat = [ones(sum(testi),1) X(testi,:)]*b_pls;
                % Calculate the Predicted REsidual Sum of Squares
                PRESS = sum((Y(testi,:)-yhat).^2,'all');
                % Calculate the total sum of squares
                TSS = sum((Y(testi,:)-mean(Y(testi,:))).^2,'all');
                Q2(p,k,c) = 1 - PRESS/TSS;

                % Calculate the mean squared error
                MSE(p,k,c) = mean((Y(testi,:)-yhat).^2,'all');
        
            end
        end
    end

    Q2 = reshape(Q2,[],mxncomp);
    Q2k = squeeze(mean(Q2));
    MSE = reshape(MSE,[],mxncomp);
    MSEk = squeeze(mean(MSE));
    [~,ncomp] = min(Q2k);
    [~,ncomp] = min(MSEk);

    figure,
    errorbar(Q2k,std(Q2)./sqrt(size(Q2,1)),'--ok')
    ylabel('Observed Predictive Relevance (Q^2)')
    xlabel('Number of components')

    figure,
    errorbar(MSEk,std(MSE)./sqrt(size(MSE,1)),'--ok')
    ylabel('Mean Squared Error')
    xlabel('Number of components')
    

end