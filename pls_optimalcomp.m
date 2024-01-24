function ncomp = pls_optimalcomp(X,Y)

    K = 5; % Number of folds for cross-validation
    % Choose upper bound on number of components to test. You could go up 
    % to rank(X), but that is very high
    ncomp = 30;
    % Instantiate matrix that we will fill with mean square errors of each fold
    % at each dimensionality
    E = NaN(K,ncomp);
    R2 = NaN(K,ncomp);
    
    % Create cross-validation partitions for K-folds
    n_subjects = size(Y,1);
    C = cvpartition(n_subjects, 'KFold', K);    
    
    for c=1:ncomp   % for each level of dimensionality
        txt = sprintf('Cross-validation of PLS-R model with %i components',c);
        fprintf(txt)
    
        for k=1:K   % for each fold
            txt2 = sprintf(': fold %i',k);
            fprintf(txt2);
            
            % Get train and test set indices for fold k
            traini = training(C,k);
            testi = test(C,k);

            % fit the PLSR model with ncomp PLS-scores on training fold
            [~,~,~,~,b_pls,~] = plsregress(X(traini,:),Y(traini,:),c);
    
            % Evaluate fit on test fold
            yhat = [ones(sum(testi),1) X(testi,:)]*b_pls;
    
            % Compute mean square error
            E(k,c) = mean((reshape(Y(testi,:),[],1)-yhat(:)).^2);
    
            fprintf(repmat('\b',1,length(txt2)))
        end
    
        fprintf(repmat('\b',1,length(txt)));
    end

    Ek = squeeze(mean(E));
    [~,ncomp] = min(Ek);

    figure,
    errorbar(Ek,std(E)./sqrt(size(E,1)),'--ok')
    ylabel('Observed Mean Square Error')
    xlabel('Number of components')

end