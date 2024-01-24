function p_mse = pls_modelvalidation(X,Y,ncomp)

    nperms = 5000;
    mse = NaN(nperms,1);
    mseperm = NaN(nperms,1);
    parfor p=1:nperms
        yhat = NaN(size(Y));
        yhatperm = yhat;
        C = cvpartition(size(Y,1),'KFold',5);
    
        % shuffle Y
        Yperm = Y(randperm(size(Y,1),size(Y,1)),:);
        for k=1:C.NumTestSets           
            % Get the training and test sets
            trn = training(C,k);
            tst = test(C,k);
        
            % Train the model with the kth partition data
            [~,~,~,~,BETA] = plsregress(X(trn,:),Y(trn,:),ncomp);
            % Fill in the predicted estimates for the kth test data
            yhat(tst,:) = [ones(sum(tst),1) X(tst,:)]*BETA;
    
            % Train the model with the kth partition data for shuffled Y
            [~,~,~,~,BETA] = plsregress(X(trn,:),Yperm(trn,:),ncomp);
            % Fill in the predicted estimates for the kth test data
            yhatperm(tst,:) = [ones(sum(tst),1) X(tst,:)]*BETA;

        end
        mse(p) = mean((yhat-Y).^2,'all');
        mseperm(p) = mean((yhatperm-Y).^2,'all');
    end
    
    rng = linspace(min([mse;mseperm]),max([mse;mseperm]),100);
    figure,histogram(mse,rng)
    hold on
    histogram(mseperm,rng)
    p_mse = sum(mseperm<mean(mse))/nperms;
    hold off
    set(gcf,'Color','w')
    legend({'Model','Null Model'})
    xlabel('Mean Squared Error')
end