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
    ncomp = 30;
    % Instantiate array that we will fill with the predictive relevance of 
    % each fold at each dimensionality
    Q2 = NaN(P,K,ncomp);
    
    
    
    for c=1:ncomp   % for each level of dimensionality
        txt = sprintf('Cross-validation of PLS-R model with %i components',c);
        fprintf(txt)
    
        parfor p=1:P
            % Create cross-validation partitions for K-folds
            n_subjects = size(Y,1);
            C = cvpartition(n_subjects, 'KFold', K);    
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
        
            end
        end
    
        fprintf(repmat('\b',1,length(txt)));
    end

    Q2 = reshape(Q2,[],c);
    Q2k = squeeze(mean(Q2));
    [~,ncomp] = min(Q2k);

    figure,
    errorbar(Q2k,std(Q2)./sqrt(size(Q2,1)),'--ok')
    ylabel('Observed Predictive Relevance (Q^2)')
    xlabel('Number of components')

end